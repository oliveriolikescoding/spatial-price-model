# -*- coding: utf-8 -*-
"""run_model.py —— 一条命令从原始表跑到三张图 + 指标表。

    python run_model.py                 # 全跑
    python run_model.py --skip-figures  # 只跑数据与模型
    python run_model.py --stages prepare,features   # 跑指定段

结束后自动做三处一致性检查，任一不过就以非零码退出。
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PY = sys.executable
BUILD = ROOT / "build"
DOCS = ROOT / "docs"

STAGES = {
    "prepare": "prepare.py",
    "features": "features.py",
    "model": "model.py",
    "figures": "figures.py",
}


def run(stage: str) -> float:
    t0 = time.time()
    print(f"\n=== {stage} · {STAGES[stage]} ===")
    r = subprocess.run([PY, str(ROOT / STAGES[stage])], cwd=ROOT)
    if r.returncode != 0:
        raise SystemExit(f"阶段 {stage} 失败（exit={r.returncode}）")
    return time.time() - t0


def check() -> None:
    import pandas as pd

    print("\n=== 一致性检查 ===")
    samples = pd.read_csv(BUILD / "samples.csv", encoding="utf-8-sig")
    feats = pd.read_csv(BUILD / "features.csv", encoding="utf-8-sig")
    oof = pd.read_csv(BUILD / "oof_predictions.csv", encoding="utf-8-sig")
    cv = json.loads((BUILD / "cv_results.json").read_text(encoding="utf-8"))
    ok = True

    c1 = len(samples) == len(feats) == len(oof)
    print(f"① 样本/特征/OOF 行数 {len(samples)} / {len(feats)} / {len(oof)} → {'一致' if c1 else '不一致'}")
    ok &= c1

    need = ["d_医疗", "d_教育", "d_购物", "d_公园", "d_地铁", "complete_score"]
    c2 = not feats[need].isna().any().any()
    print(f"② 可达性特征无缺失 → {'通过' if c2 else '有缺失'}")
    ok &= c2

    g = {(r["scheme"], r["model"]) for r in cv["results"]}
    c3 = len(g) == 9 and all(("group_district", m) in g for m in ["dummy", "linear", "hgb"])
    print(f"③ 3 种切分 × 3 个模型 = {len(g)} 组结果 → {'齐' if c3 else '缺'}")
    ok &= c3

    figs = ["fig_cv_gap.png", "fig_residual_map.png", "fig_feature_importance.png"]
    c4 = all((DOCS / f).exists() for f in figs)
    print(f"④ 三张图齐 → {'是' if c4 else '缺'}")
    ok &= c4

    gap = cv["cv_gap"]["hgb"]
    print(f"\n关键数：随机 CV R²={gap['r2_random']:+.3f} / 按区县 {gap['r2_group_district']:+.3f}"
          f" → 虚高 {gap['inflation_vs_district'] * 100:.1f} 个百分点")
    print(f"残差 Moran's I={cv['moran_residual']['I']} (p={cv['moran_residual']['p_perm']})")
    if not ok:
        raise SystemExit("一致性检查未通过")
    print("全部通过。")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-figures", action="store_true")
    ap.add_argument("--stages", default="")
    a = ap.parse_args()
    order = [s for s in ["prepare", "features", "model", "figures"] if s in STAGES]
    if a.stages:
        order = [s.strip() for s in a.stages.split(",") if s.strip() in STAGES]
    if a.skip_figures and "figures" in order:
        order.remove("figures")

    t0 = time.time()
    times = {}
    for s in order:
        times[s] = run(s)
    total = time.time() - t0
    print("\n=== 耗时 ===")
    for k, v in times.items():
        print(f"  {k:9s} {v:6.1f} s")
    print(f"  合计      {total:6.1f} s")
    if "figures" in order:
        check()


if __name__ == "__main__":
    main()
