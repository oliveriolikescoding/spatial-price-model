# -*- coding: utf-8 -*-
"""features.py —— 把「一个小区」变成一行特征。

分四组，每组在 build/feature_manifest.csv 里写明来源与时点：
  A 可达性  来自 W10 单元表（到五类设施最近距离）+ 到天府广场直线距离
  B 密度    社区邻域缓冲内的设施/地铁/公园计数（来自 W10 设施点与站点）
  C 形态    住宅面面积、栋数、户数、楼龄（as-of 价格年份）
  D 配套    物业费、W10 完备度、建筑类型

目标变量 y 在 model.py 里算（区内 log 单价的 z-score），不在这里出现，
这样 features.csv 可以原样给别的项目复用。
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
DB = ROOT.parent / "geo-etl-pipeline" / "data" / "chengdu.db"
BUILD = ROOT / "build"
BUILD.mkdir(exist_ok=True)

PRICE_YEAR = 2017          # 价格数据的快照年份 2017-08-30（见 docs/data_manifest.md），楼龄以它为基准算
CENTER = (30.6572, 104.0657)  # 天府广场
R_EARTH = 6371000.0


def haversine(lat1, lon1, lat2, lon2):
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp = p2 - p1
    dl = np.radians(lon2 - lon1)
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * R_EARTH * np.arcsin(np.sqrt(a))


def km_xy(lat0, lon0, lats, lons):
    """小范围内把经纬度换成公里平面坐标（等距圆柱近似），用于邻域计数。"""
    x = (lons - lon0) * 111.320 * np.cos(np.radians(lat0))
    y = (lats - lat0) * 110.570
    return x, y


def main() -> None:
    samples = pd.read_csv(BUILD / "samples.csv", encoding="utf-8-sig")
    con = sqlite3.connect(DB)
    units = pd.read_sql(
        "select osm_type, osm_id, 完备度, dist_医疗, dist_教育, dist_购物, dist_公园, dist_地铁 "
        "from units",
        con,
    )
    fac = pd.read_sql("select 类别, 经度, 纬度 from facilities", con)
    sta = pd.read_sql("select 经度, 纬度 from stations", con)
    con.close()

    units["key"] = units["osm_type"] + "/" + units["osm_id"].astype(str)
    samples["key"] = samples["osm_type"] + "/" + samples["osm_id"].astype(str)
    f = samples.merge(units.drop(columns=["osm_type", "osm_id"]), on="key", how="left")
    assert f["dist_医疗"].notna().all(), "有样本没接上 W10 单元表"

    # ---- A 可达性 -----------------------------------------------------------
    f["d_中心"] = haversine(f["lat"], f["lon"], CENTER[0], CENTER[1])
    for src, dst in [("dist_医疗", "d_医疗"), ("dist_教育", "d_教育"),
                     ("dist_购物", "d_购物"), ("dist_公园", "d_公园"),
                     ("dist_地铁", "d_地铁")]:
        f[dst] = f[src].astype(float)

    # ---- B 密度 -------------------------------------------------------------
    R500, R1000 = 0.5, 1.0
    n_fac_500, n_fac_1000, n_sub_1000, n_park_1000 = [], [], [], []
    for row in f.itertuples(index=False):
        x, y = km_xy(row.lat, row.lon, fac["纬度"].values, fac["经度"].values)
        d = np.sqrt(x ** 2 + y ** 2)
        n_fac_500.append(int((d <= R500).sum()))
        n_fac_1000.append(int((d <= R1000).sum()))
        xs, ys = km_xy(row.lat, row.lon, sta["纬度"].values, sta["经度"].values)
        n_sub_1000.append(int((np.sqrt(xs ** 2 + ys ** 2) <= R1000).sum()))
        p = fac["类别"].values == "公园"
        n_park_1000.append(int((d[p] <= R1000).sum()))
    f["n_fac_500"] = n_fac_500
    f["n_fac_1000"] = n_fac_1000
    f["n_sub_1000"] = n_sub_1000
    f["n_park_1000"] = n_park_1000

    # ---- C 形态 -------------------------------------------------------------
    f["log_area"] = np.log1p(f["unit_area_m2"].clip(lower=1))
    f["log_households"] = np.log1p(f["n_households"].fillna(f["n_households"].median()))
    f["n_buildings"] = f["n_buildings"].fillna(f["n_buildings"].median())
    f["aging"] = (PRICE_YEAR - f["build_year"]).clip(lower=0)
    f["has_year"] = f["build_year"].notna().astype(int)
    f["aging"] = f["aging"].fillna(f["aging"].median())

    # ---- D 配套 -------------------------------------------------------------
    f["fee_yuan"] = f["fee_yuan"].fillna(f["fee_yuan"].median())
    f["complete_score"] = f["完备度"].astype(float)

    # 建筑类型 one-hot（只保留样本量 ≥10 的类别，其余并入 other）
    bt = f["build_type"].fillna("unknown")
    keep = [k for k, v in bt.value_counts().items() if v >= 10 and k != "unknown"]
    for k in keep:
        f[f"bt_{k}"] = (bt == k).astype(int)
    f["bt_other"] = (~bt.isin(keep)).astype(int)

    num_cols = ["d_医疗", "d_教育", "d_购物", "d_公园", "d_地铁", "d_中心",
                "n_fac_500", "n_fac_1000", "n_sub_1000", "n_park_1000",
                "log_area", "log_households", "n_buildings", "aging", "has_year",
                "fee_yuan", "complete_score"] + [f"bt_{k}" for k in keep] + ["bt_other"]
    out = f[["key", "osm_type", "osm_id", "小区名称", "区县", "片区", "lat", "lon",
             "price_per_sqm"] + num_cols].copy()
    # 目标变量原料留在这里（price_per_sqm），z-score 在 model.py 里按 CV 折内算
    out.to_csv(BUILD / "features.csv", index=False, encoding="utf-8-sig")

    manifest = pd.DataFrame(
        [
            ("d_医疗", "A 可达性", "W10 units.dist_医疗", "2026-11 OSM"),
            ("d_教育", "A 可达性", "W10 units.dist_教育", "2026-11 OSM"),
            ("d_购物", "A 可达性", "W10 units.dist_购物", "2026-11 OSM"),
            ("d_公园", "A 可达性", "W10 units.dist_公园", "2026-11 OSM"),
            ("d_地铁", "A 可达性", "W10 units.dist_地铁", "2026-11 OSM"),
            ("d_中心", "A 可达性", "样本点 → 天府广场 haversine", "2026-11 OSM"),
            ("n_fac_500", "B 密度", "500 m 内设施计数", "2026-11 OSM"),
            ("n_fac_1000", "B 密度", "1000 m 内设施计数", "2026-11 OSM"),
            ("n_sub_1000", "B 密度", "1000 m 内地铁站计数", "2026-11 OSM"),
            ("n_park_1000", "B 密度", "1000 m 内公园计数", "2026-11 OSM"),
            ("log_area", "C 形态", "住宅面面积取对数", "2026-11 OSM"),
            ("log_households", "C 形态", "链家 总户数 取对数", "2018 链家"),
            ("n_buildings", "C 形态", "链家 总栋数", "2018 链家"),
            ("aging", "C 形态", "2018 − 建筑年代", "2018 链家"),
            ("has_year", "C 形态", "建筑年代是否有值", "2018 链家"),
            ("fee_yuan", "D 配套", "链家 物业费（区间取中值）", "2018 链家"),
            ("complete_score", "D 配套", "W10 units.完备度（0–5）", "2026-11 OSM"),
        ]
        + [(c, "D 配套", "链家 建筑类型 one-hot", "2018 链家")
           for c in num_cols if c.startswith("bt_")],
        columns=["feature", "group", "definition", "asof"],
    )
    manifest.to_csv(BUILD / "feature_manifest.csv", index=False, encoding="utf-8-sig")

    # 共线性体检：|r| > 0.9 的组合单独记账
    corr = out[num_cols].corr().abs()
    pairs = [
        {"a": num_cols[i], "b": num_cols[j], "r": round(float(corr.iloc[i, j]), 3)}
        for i in range(len(num_cols)) for j in range(i + 1, len(num_cols))
        if corr.iloc[i, j] > 0.9
    ]
    (BUILD / "feature_log.json").write_text(
        json.dumps(
            {
                "n_rows": int(len(out)),
                "n_features": len(num_cols),
                "features": num_cols,
                "high_corr_pairs": pairs,
                "build_type_kept": keep,
                "missing": out[num_cols].isna().sum().to_dict(),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"features.csv {len(out)} 行 × {len(num_cols)} 特征；高相关对 {len(pairs)} 组："
          f"{[(p['a'], p['b'], p['r']) for p in pairs]}")


if __name__ == "__main__":
    main()
