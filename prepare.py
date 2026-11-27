# -*- coding: utf-8 -*-
"""prepare.py —— 把链家小区挂牌表接上真实坐标。

思路（也是本项目唯一一处"外部数据接入"）：
  链家小区表有价格、无坐标；W10 的 Overpass 住宅面有名字、有质心坐标。
  → 用「小区名」做键，把价格挂到 OSM 住宅面的质心上。
  这不是 geocoding，是**名称关联**：命中率不高但坐标是真几何质心，不是猜的。

输入：raw/lianjia_cd_xiaoqu.csv（公开历史数据集，见 docs/data_manifest.md）
      ../../geo-etl-pipeline/data/chengdu.db（W10 成果，units 表自带坐标、区县、可达性）
输出：build/samples.csv、build/match_log.json
"""
from __future__ import annotations

import json
import re
import sqlite3
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "raw" / "lianjia_cd_xiaoqu.csv"
DB = ROOT.parent / "geo-etl-pipeline" / "data" / "chengdu.db"
BUILD = ROOT / "build"
BUILD.mkdir(exist_ok=True)

# 区内 z-score 的样本下限：区县内样本少于 20 个，该区的「区内相对价」不可信
MIN_DISTRICT_CT = 20

# 名称归一：去括号内容、去空格间隔号、去通用后缀（用于名称关联，不用于展示）
PAREN = re.compile(r"[（(].*?[）)]")
SPACE = re.compile(r"[\s·・\-—_]+")
SUFFIX = re.compile(
    r"(花园|家园|公寓|公馆|华庭|小区|社区|新城|广场|国际|苑|园|城|府|湾|庭|轩|阁|大厦|雅居|名邸)$"
)


def norm_name(s: str) -> str:
    s = PAREN.sub("", str(s)).strip()
    s = SPACE.sub("", s)
    return s


def core_name(s: str) -> str:
    s = norm_name(s)
    return SUFFIX.sub("", s) or s


def norm_district(s: str) -> str:
    s = str(s).strip()
    return re.sub(r"(区|县|市)$", "", s)


def num(series: pd.Series) -> pd.Series:
    """从「15597」「2.5至8元/平米/月」这类脏串里取出数值；区间取中值。"""
    def one(v):
        if pd.isna(v):
            return None
        t = str(v)
        if "暂无" in t or t.strip() in {"", "-"}:
            return None
        nums = re.findall(r"\d+(?:\.\d+)?", t)
        if not nums:
            return None
        vals = [float(x) for x in nums]
        return sum(vals) / len(vals) if len(vals) > 1 else vals[0]

    return series.map(one)


def main() -> None:
    log: dict = {}
    if not RAW.exists():
        raise SystemExit(f"缺原始件：{RAW}（先按 docs/data_manifest.md 下载）")

    raw = pd.read_csv(RAW, encoding="utf-8-sig")
    log["raw_rows"] = int(len(raw))
    log["raw_columns"] = list(raw.columns)

    # ---- 1. 价格与属性解析 --------------------------------------------------
    df = pd.DataFrame(
        {
            "xiaoqu_id": raw["ID"].astype(str),
            "小区名称": raw["小区名称"].astype(str).str.strip(),
            "区县_csv": raw["区域"].astype(str).str.strip(),
            "片区": raw["片区"].astype(str).str.strip(),
            "price_per_sqm": num(raw["参考均价"]),
            "build_year": num(raw["建筑年代"]),
            "n_buildings": num(raw["总栋数"]),
            "n_households": num(raw["总户数"]),
            "fee_yuan": num(raw["物业费"]),
            "build_type": raw["建筑类型"].astype(str).str.strip(),
            "地址": raw["地址"].astype(str).str.strip(),
        }
    )
    log["price_missing"] = int(df["price_per_sqm"].isna().sum())
    df = df[df["price_per_sqm"].notna()].copy()
    log["after_price_drop"] = int(len(df))

    # ---- 2. OSM 住宅面（带名字的才有资格当被关联对象） ----------------------
    con = sqlite3.connect(DB)
    units = pd.read_sql("select * from units", con)
    con.close()
    units["区县_n"] = units["区县"].map(norm_district)
    named = units[units["name_src"] != "generated"].copy()
    named["core"] = named["单元名"].map(core_name)
    log["osm_units"] = int(len(units))
    log["osm_named_units"] = int(len(named))

    index: dict[str, pd.DataFrame] = {
        k: g for k, g in named.groupby("core")
    }
    # 第二轮用的「包含匹配」候选：core 长度 ≥4 才参与，防「一期」「南区」这类碎片乱挂
    long_core = named[named["core"].str.len() >= 4].copy()

    def contain_hit(name: str, district: str):
        c = core_name(name)
        if len(c) < 4:
            return None
        dn = norm_district(district)
        cand = long_core[
            long_core["core"].map(lambda o: c in o or o in c)
            & (long_core["区县_n"] == dn)
        ]
        # 唯一才认：多个候选说明名字太泛，宁可不挂
        return cand.iloc[0] if len(cand) == 1 else None

    # ---- 3. 名称关联 + 区县消歧 ---------------------------------------------
    rows, dropped = [], {"no_name_hit": 0, "ambiguous_same_district": 0,
                         "ambiguous_cross_district": 0}
    pass2_hits = 0
    for rec in df.itertuples(index=False):
        core = core_name(rec.小区名称)
        cand = index.get(core)
        if cand is None:
            # 退一步：全名归一后再试一次（后缀剥过头的兜底）
            cand = named[named["单元名"].map(norm_name) == norm_name(rec.小区名称)]
        if cand is None or len(cand) == 0:
            # 第二轮：同一区县内的「名称包含」唯一命中
            loose = contain_hit(rec.小区名称, rec.区县_csv)
            if loose is None:
                dropped["no_name_hit"] += 1
                continue
            cand = pd.DataFrame([loose])
            pass2_hits += 1
        if len(cand) > 1:
            same = cand[cand["区县_n"] == norm_district(rec.区县_csv)]
            if len(same) == 1:
                cand = same
            elif len(same) == 0:
                dropped["ambiguous_cross_district"] += 1
                continue
            else:
                dropped["ambiguous_same_district"] += 1
                continue
        u = cand.iloc[0]
        rows.append(
            {
                "xiaoqu_id": rec.xiaoqu_id,
                "小区名称": rec.小区名称,
                "osm_type": u["osm_type"],
                "osm_id": u["osm_id"],
                "单元名": u["单元名"],
                "区县": u["区县"],
                "片区": rec.片区,
                "lat": float(u["质心纬度"]),
                "lon": float(u["质心经度"]),
                "price_per_sqm": float(rec.price_per_sqm),
                "build_year": rec.build_year,
                "n_buildings": rec.n_buildings,
                "n_households": rec.n_households,
                "fee_yuan": rec.fee_yuan,
                "build_type": rec.build_type,
                "地址": rec.地址,
                "unit_area_m2": float(u["面积_㎡"]),
                "complete_score": float(u["完备度"]),
            }
        )
    matched = pd.DataFrame(rows)
    log["name_match_rows"] = int(len(matched))
    log["pass2_containment_hits"] = int(pass2_hits)
    log["drop_detail"] = dropped

    # ---- 4. 异常值剔除（记账） ---------------------------------------------
    before = len(matched)
    q1, q99 = matched["price_per_sqm"].quantile([0.01, 0.99])
    mask = matched["price_per_sqm"].between(max(q1, 2000), min(q99, 150000))
    matched = matched[mask].copy()
    log["price_outlier_dropped"] = int(before - len(matched))
    log["iqr_band"] = [round(float(q1), 0), round(float(q99), 0)]

    # 同一 OSM 单元被多个小区命中 → 取中位价（单元为分析单元）
    g = matched.groupby(["osm_type", "osm_id"], as_index=False).agg(
        price_per_sqm=("price_per_sqm", "median"),
        n_obs=("price_per_sqm", "size"),
        小区名称=("小区名称", lambda s: s.mode().iat[0] if len(s.mode()) else s.iat[0]),
        区县=("区县", "first"),
        片区=("片区", lambda s: s.mode().iat[0] if len(s.mode()) else s.iat[0]),
        lat=("lat", "first"),
        lon=("lon", "first"),
        build_year=("build_year", "median"),
        n_buildings=("n_buildings", "median"),
        n_households=("n_households", "median"),
        fee_yuan=("fee_yuan", "median"),
        build_type=("build_type", lambda s: s.mode().iat[0] if len(s.mode()) else "unknown"),
        unit_area_m2=("unit_area_m2", "first"),
        complete_score=("complete_score", "first"),
    )
    n_collapse = int(len(matched) - len(g))
    log["rows_before_unit_dedup"] = int(len(matched))
    log["rows_dropped_by_unit_dedup"] = n_collapse
    log["final_samples"] = int(len(g))
    log["district_count"] = int(g["区县"].nunique())
    log["area_count"] = int(g["片区"].nunique())
    log["district_dist"] = g["区县"].value_counts().to_dict()

    g = g.sort_values("price_per_sqm").reset_index(drop=True)

    # 区内 z-score 需要区内样本量撑着：n<20 的区县，z 本身不可信，整区剔除（记账）
    cnt = g["区县"].value_counts()
    thin = cnt[cnt < MIN_DISTRICT_CT].index.tolist()
    log["districts_below_min"] = {k: int(cnt[k]) for k in thin}
    log["rows_dropped_thin_district"] = int(g["区县"].isin(thin).sum())
    g = g[~g["区县"].isin(thin)].reset_index(drop=True)
    log["final_samples"] = int(len(g))
    log["district_count"] = int(g["区县"].nunique())
    log["area_count"] = int(g["片区"].nunique())
    log["district_dist"] = g["区县"].value_counts().to_dict()

    g.to_csv(BUILD / "samples.csv", index=False, encoding="utf-8-sig")
    (BUILD / "match_log.json").write_text(
        json.dumps(log, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"原始 {log['raw_rows']} 行 → 有价 {log['after_price_drop']} → "
          f"名称命中 {log['name_match_rows']} → 剔除异常后去重 {log['final_samples']} 个单元样本")
    print(f"覆盖 {log['district_count']} 个区县、{log['area_count']} 个片区")
    print("命中率：", f"{log['name_match_rows'] / log['after_price_drop']:.1%}")
    print("未命中原因：", dropped)


if __name__ == "__main__":
    main()
