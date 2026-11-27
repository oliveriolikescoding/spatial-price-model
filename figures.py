# -*- coding: utf-8 -*-
"""figures.py —— 三张图，一张说方法、一张说漏在哪、一张说谁在起作用。

制图口径（与 W10 一致，RV-15 判为信息类 → 专业制图风）：
    中文宋体 SimSun + 西文 Times New Roman；色阶语义化、不彩虹；
    残差用双向发散色阶（0 居中，暖=低估）；八件套齐。
"""
from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager
from matplotlib.colors import TwoSlopeNorm

ROOT = Path(__file__).resolve().parent
BUILD = ROOT / "build"
DOCS = ROOT / "docs"
DOCS.mkdir(exist_ok=True)
DISTRICTS = ROOT.parent / "geo-etl-pipeline" / "raw" / "districts_chengdu_gaode.json"

MAP_DATE = "2026-11-27"
COOL = "#2F5C8A"
WARM = "#B4483C"

# 特征 → 中文标签（图面要能读懂，别把 d_医疗 直接扔上去）
LABEL = {
    "d_医疗": "到最近医疗设施", "d_教育": "到最近教育设施", "d_购物": "到最近购物点",
    "d_公园": "到最近公园", "d_地铁": "到最近地铁站", "d_中心": "到天府广场",
    "n_fac_500": "500 m 设施数", "n_fac_1000": "1 km 设施数",
    "n_sub_1000": "1 km 地铁站数", "n_park_1000": "1 km 公园数",
    "log_area": "住宅片区面积", "log_households": "户数（对数）",
    "n_buildings": "楼栋数", "aging": "楼龄（至 2017）", "has_year": "有建筑年代",
    "fee_yuan": "物业费", "complete_score": "W10 完备度",
}


def setup_fonts() -> bool:
    ok = True
    for fam in ["SimSun", "Times New Roman"]:
        if not any(f.name == fam for f in font_manager.fontManager.ttflist):
            ok = False
    plt.rcParams["font.family"] = ["Times New Roman", "SimSun"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.dpi"] = 110
    return ok


def add_caption(fig, no: int, title: str, notes: list[str], top: float,
                step: float = 0.030) -> None:
    """论文式图注：图号 + 题名居中放在图下，其后接说明行（小字、灰色）。"""
    fig.text(0.5, top, f"图 {no}　{title}", ha="center", va="top",
             fontsize=12.5, fontweight="bold", color="#1a1a1a")
    for i, n in enumerate(notes):
        fig.text(0.5, top - 0.038 - i * step, n, ha="center", va="top",
                 fontsize=8.4, color="#555555")


def load_all():
    cv = json.loads((BUILD / "cv_results.json").read_text(encoding="utf-8"))
    oof = pd.read_csv(BUILD / "oof_predictions.csv", encoding="utf-8-sig")
    return cv, oof


# ---------------------------------------------------------------------------
def fig_cv_gap(cv: dict) -> Path:
    res = cv["results"]
    schemes = [("random", "随机 5 折"), ("group_area", "按片区分组"), ("group_district", "按区县分组")]
    models = [("linear", "线性（岭回归）"), ("hgb", "梯度提升树")]
    colors = {0: "#C9D9E6", 1: "#5C86AE"}

    fig, ax = plt.subplots(figsize=(12.6, 7.0))
    x = np.arange(len(schemes))
    width = 0.34
    for j, (mkey, mlabel) in enumerate(models):
        vals = [next(r["r2"] for r in res if r["scheme"] == s and r["model"] == mkey)
                for s, _ in schemes]
        ax.bar(x + (j - 0.5) * width, vals, width * 0.94,
               color=colors[j], edgecolor="#3A3A3A", linewidth=0.6, label=mlabel)
        for xi, v in zip(x, vals):
            ax.text(xi + (j - 0.5) * width, v + (0.02 if v >= 0 else -0.05),
                    f"{v:+.3f}", ha="center",
                    va="bottom" if v >= 0 else "top", fontsize=9.5)

    # 0 线 = 区内均值基线
    ax.axhline(0, color="#444444", lw=1.0, ls="-")
    ax.text(len(schemes) - 0.42, 0.012, "", fontsize=9.5)

    gap = cv["cv_gap"]["hgb"]
    # 用双头箭头 + 文字标出虚高量，放在两组柱子之间的空档，避免压住柱子和图例
    xr = 1.62
    ax.annotate("", xy=(xr, gap["r2_random"]), xytext=(xr, gap["r2_group_district"]),
                arrowprops=dict(arrowstyle="<->", color=WARM, lw=1.5))
    ax.text(xr + 0.05, (gap["r2_random"] + gap["r2_group_district"]) / 2,
            f"虚高 {gap['inflation_vs_district'] * 100:.1f} 个百分点",
            fontsize=11.5, color=WARM, ha="left", va="center", fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels([s[1] for s in schemes], fontsize=11.5)
    ax.set_ylabel("R²（解释区内价差的份额）", fontsize=11.5)
    ax.set_ylim(-0.75, 0.60)
    ax.set_xlim(-0.62, 2.62)
    ax.axhspan(-0.75, 0, color="#F2F2F2", zorder=0)
    ax.text(0.012, 0.055, "阴影区为负值：模型不如「直接猜区内均值」",
            fontsize=9, color="#777777", transform=ax.transAxes)
    ax.grid(axis="y", ls=":", color="#CCCCCC", lw=0.7)
    ax.set_axisbelow(True)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    from matplotlib.lines import Line2D
    handles, labels = ax.get_legend_handles_labels()
    handles.append(Line2D([0], [0], color="#444444", lw=1.2))
    labels.append("基线 R² = 0（区内均值）")
    ax.legend(handles, labels, loc="upper right", frameon=False, fontsize=11)

    fig.subplots_adjust(left=0.09, right=0.985, top=0.985, bottom=0.225)
    add_caption(
        fig, 1, "交叉验证方式对模型精度的影响",
        ["同一组特征与超参数，只更换切分方式；基线为区县内均值（R² = 0）。"
         "样本 525 个住宅小区，5 个中心城区，26 个特征。",
         "数据：链家小区挂牌（2017-08）＋ OpenStreetMap 设施与住宅面（2026-11，ODbL）。"
         "指标为按测试集样本数加权的 5 折均值。制图：Matplotlib，宋体 + Times New Roman。"],
        top=0.168)
    out = DOCS / "fig_cv_gap.png"
    fig.savefig(out, dpi=300, bbox_inches=None)
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
def fig_residual_map(oof: pd.DataFrame) -> Path:
    dg = gpd.read_file(DISTRICTS)
    pts = gpd.GeoDataFrame(
        oof, geometry=gpd.points_from_xy(oof["lon"], oof["lat"]), crs="EPSG:4326"
    )
    q = np.nanpercentile(np.abs(pts["resid_pct"]), 95)
    norm = TwoSlopeNorm(vmin=-q, vcenter=0, vmax=q)

    fig, ax = plt.subplots(figsize=(10.6, 8.6))
    dg.plot(ax=ax, facecolor="#EFEFEF", edgecolor="#8A8A8A", linewidth=0.9, zorder=1)
    sc = ax.scatter(pts.geometry.x, pts.geometry.y, c=pts["resid_pct"],
                    cmap="RdBu_r", norm=norm, s=16, edgecolor="#333333",
                    linewidth=0.25, zorder=2)
    cb = fig.colorbar(sc, ax=ax, shrink=0.62, pad=0.02)
    cb.set_label("预测偏差（%）：正值＝低估（实际更贵）", fontsize=10.5)
    cb.ax.tick_params(labelsize=9)

    # 区县名：只标有样本的区，位置取该区样本点的重心（district representative_point
    # 会落在数据范围之外，导致标签被裁掉）
    lab = pts.groupby("区县").agg(x=("lon", "mean"), y=("lat", "mean"))
    for dname, r in lab.iterrows():
        ax.text(r["x"], r["y"], dname, fontsize=10.5, color="#666666", ha="center",
                va="center", alpha=0.9, zorder=3,
                bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="none", alpha=0.55))

    # 标出最偏的两个片区（取样本 ≥8 的片区中位数，避免单个极端点代表整个片区）
    area = pts.groupby("片区")["resid_pct"].agg(["median", "size"]).query("size >= 8")
    top_under = area["median"].idxmax()
    top_over = area["median"].idxmin()
    for name_, val, dy, dx in [(top_under, area.loc[top_under, "median"], 1, -0.008),
                               (top_over, area.loc[top_over, "median"], -1, 0.008)]:
        sub = pts[pts["片区"] == name_]
        cx, cy = sub.geometry.x.mean(), sub.geometry.y.mean()
        ax.annotate(f"{name_}\n中位 {val:+.0f}%", xy=(cx, cy),
                    xytext=(cx + dx, cy + dy * 0.075),
                    fontsize=10, ha="center",
                    color=WARM if val > 0 else COOL, fontweight="bold",
                    arrowprops=dict(arrowstyle="-", color="#666666", lw=0.8),
                    bbox=dict(boxstyle="round,pad=0.22", fc="white", ec="#CCCCCC", alpha=0.9))

    b = pts.total_bounds
    ax.set_xlim(b[0] - 0.015, b[2] + 0.015)
    ax.set_ylim(b[1] - 0.015, b[3] + 0.012)
    ax.set_xlabel("经度（WGS84）", fontsize=10.5)
    ax.set_ylabel("纬度（WGS84）", fontsize=10.5)
    ax.set_aspect("equal")
    ax.grid(ls=":", color="#DDDDDD", lw=0.6)
    ax.set_axisbelow(True)

    # 比例尺（按真实米数换算）
    xmin, xmax = ax.get_xlim()
    lat0 = np.mean(ax.get_ylim())
    km5 = 5000 / (111320 * np.cos(np.radians(lat0)))
    x0, y0 = xmin + (xmax - xmin) * 0.03, ax.get_ylim()[0] + 0.018
    ax.plot([x0, x0 + km5], [y0, y0], color="#333333", lw=2.2, solid_capstyle="butt")
    for f in (0, 0.5, 1):
        ax.plot([x0 + km5 * f, x0 + km5 * f], [y0, y0 + 0.004], color="#333333", lw=2.2)
    ax.text(x0 + km5 / 2, y0 + 0.007, "5 km", fontsize=9.5, ha="center")

    # 指北：箭头放右侧，字母放在箭头左边，避免压在箭杆上
    ax.annotate("", xy=(xmax - 0.03, ax.get_ylim()[1] - 0.008),
                xytext=(xmax - 0.03, ax.get_ylim()[1] - 0.055),
                arrowprops=dict(arrowstyle="-|>", color="#333333", lw=1.6))
    ax.text(xmax - 0.042, ax.get_ylim()[1] - 0.032, "N", fontsize=12.5,
            ha="right", va="center", fontweight="bold")

    moran = {"I": None}
    try:
        moran = json.loads((BUILD / "cv_results.json").read_text(encoding="utf-8"))["moran_residual"]
    except Exception:
        pass

    fig.subplots_adjust(left=0.02, right=0.985, top=0.985, bottom=0.235)
    add_caption(
        fig, 2, "模型残差的空间分布",
        ["梯度提升树，按区县分组 5 折的 out-of-fold 预测。残差 = 区内实际 log 单价与预测之差，"
         "按该区 log 标准差还原为百分比。",
         f"Moran's I = {moran['I']}（p = {moran['p_perm']}，8 邻域 KNN 权重，999 次置换）。"
         "暖色为低估（实际更贵），冷色为高估；色阶截断于 ±95 分位。",
         "片区标注取样本数 ≥ 8 的片区中位数。投影 WGS84，比例尺按 30.7°N 换算。"
         "数据：链家挂牌（2017-08）＋ OSM（2026-11，ODbL）。制图 2026-11-27。"],
        top=0.130, step=0.032)
    out = DOCS / "fig_residual_map.png"
    fig.savefig(out, dpi=300, bbox_inches=None)
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
def fig_importance(cv: dict) -> Path:
    imp = cv["importance"]
    items = list(imp.items())[:12][::-1]
    names = [LABEL.get(k, k) for k, _ in items]
    means = [v["mean"] for _, v in items]
    sds = [v["sd_across_folds"] for _, v in items]

    man = pd.read_csv(BUILD / "feature_manifest.csv", encoding="utf-8-sig")
    grp = dict(zip(man["feature"], man["group"]))
    colors = [COOL if grp.get(k, "").startswith(("A", "B")) else "#9A9A9A" for k, _ in items]

    fig, ax = plt.subplots(figsize=(11.6, 7.2))
    ax.barh(np.arange(len(names)), means, xerr=sds, color=colors,
            edgecolor="#3A3A3A", linewidth=0.6, error_kw=dict(ecolor="#555555", lw=0.9, capsize=3))
    ax.set_yticks(np.arange(len(names)))
    ax.set_yticklabels(names, fontsize=11)
    for i, (m, s) in enumerate(zip(means, sds)):
        ax.text(m + s + 0.012, i, f"{m:.3f}", va="center", fontsize=9.5)
    ax.axvline(0, color="#444444", lw=0.9)
    ax.set_xlabel("置换重要性（打乱该特征后 R² 的下降量）", fontsize=11.5)
    ax.set_xlim(0, max(m + s for m, s in zip(means, sds)) * 1.16)
    ax.grid(axis="x", ls=":", color="#CCCCCC", lw=0.7)
    ax.set_axisbelow(True)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)

    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=COOL, label="空间类（可达性 + 密度）"),
                       Patch(color="#9A9A9A", label="属性类（形态 + 配套）")],
              loc="lower right", frameon=False, fontsize=10.5)

    fig.subplots_adjust(left=0.16, right=0.985, top=0.985, bottom=0.255)
    add_caption(
        fig, 3, "各特征的置换重要性",
        ["梯度提升树，按区县分组 5 折，在测试折上各做 15 次置换后取均值；误差棒为折间标准差。"
         "置换成负值表示打乱该特征后模型反而略好，属噪声。",
         "数据：链家小区挂牌（2017-08）＋ OSM（2026-11，ODbL）。配色区分空间类（可达性、密度）"
         "与属性类（形态、配套）。置换重要性衡量相关性，不代表因果。制图 2026-11-27。"],
        top=0.168, step=0.032)
    out = DOCS / "fig_feature_importance.png"
    fig.savefig(out, dpi=300, bbox_inches=None)
    plt.close(fig)
    return out


def main() -> None:
    font_ok = setup_fonts()
    cv, oof = load_all()
    for p in [fig_cv_gap(cv), fig_residual_map(oof), fig_importance(cv)]:
        print(f"出图 {p.name} {p.stat().st_size / 1024:.0f} KB")
    if not font_ok:
        print("警告：字体未全部命中")


if __name__ == "__main__":
    main()

