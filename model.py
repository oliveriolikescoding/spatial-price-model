# -*- coding: utf-8 -*-
"""model.py —— 三个模型 × 三套切分法，同一份特征、同一套流程。

为什么目标变量是「区内相对价」：
    绝对单价里最大的一块信息是「你在哪个区」——那不是空间分析，那是查表。
    把每个区县的 log 单价减去本区均值、除以本区标准差，模型要解释的就只剩
    「同一个区里，为什么这个小区比那个贵」。可达性、密度、形态才能说话。

为什么切分法跑三套：
    随机 KFold 会让相邻小区同时进训练集和测试集，空间自相关把分数灌高。
    按片区、按区县分组的两套切分，才是「去一个没见过的街区预测」的真实难度。
    三套的差值，就是本项目要报告的那个数。

防泄漏：区县的均值/标准差只用训练折估计，再套到测试折上。
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
BUILD = ROOT / "build"
RANDOM = 42


def fit_stats(y_log: np.ndarray, district: np.ndarray) -> dict:
    per = {}
    for d in np.unique(district):
        m = district == d
        per[d] = (float(y_log[m].mean()), float(y_log[m].std(ddof=0) + 1e-9))
    return {"per_district": per,
            "global": (float(y_log.mean()), float(y_log.std(ddof=0) + 1e-9))}


def apply_z(y_log: np.ndarray, district: np.ndarray, stats: dict) -> np.ndarray:
    out = np.empty(len(y_log), dtype=float)
    for i, (v, d) in enumerate(zip(y_log, district)):
        mu, sd = stats["per_district"].get(d, stats["global"])
        out[i] = (v - mu) / sd
    return out


def build_models() -> dict:
    return {
        "dummy": DummyRegressor(strategy="mean"),
        # 26 个特征里有成组共线（五个距离、一堆 one-hot），普通 OLS 在某一折上会数值爆炸
        # （实测出现过 -4.4e24 的折 R²），因此线性这一支用岭回归：仍是线性模型，只是加了
        # 一个 alpha=1 的惩罚，让共线方向上的系数取最小方差解。
        "linear": Pipeline([("sc", StandardScaler()), ("ridge", Ridge(alpha=1.0))]),
        "hgb": HistGradientBoostingRegressor(
            max_iter=400, learning_rate=0.05, min_samples_leaf=15,
            l2_regularization=0.1, random_state=RANDOM,
        ),
    }


def scheme_splits(df: pd.DataFrame) -> dict:
    return {
        "random": list(KFold(n_splits=5, shuffle=True, random_state=RANDOM).split(df)),
        "group_area": list(GroupKFold(n_splits=5).split(df, groups=df["片区"])),
        "group_district": list(GroupKFold(n_splits=5).split(df, groups=df["区县"])),
    }


def run_cv(df, X, y_log, model_name, model, scheme_name, splits):
    """逐折评估。

    注意一个容易踩的坑：各折的 z 基准不同（每折只用训练集估计区县均值/标准差），
    把所有折的 OOF 拼在一起算一个全局 R² 是错的——实测线性模型会因此掉到 -0.49，
    纯属基准错位而非模型差。这里一律用「按测试集样本数加权的折均值」，并在
    per_fold 里留全部折的分项，谁都能复核。
    """
    oof = np.full(len(df), np.nan)        # 预测值（各折自己的 z 尺度）
    oof_yz = np.full(len(df), np.nan)     # 同尺度的真实值
    oof_sd = np.full(len(df), np.nan)     # 该折该区县的 log 价格标准差，用于还原元/㎡
    per_fold = []
    for k, (tr, te) in enumerate(splits):
        stats = fit_stats(y_log[tr], df["区县"].values[tr])
        ytr_z = apply_z(y_log[tr], df["区县"].values[tr], stats)
        yte_z = apply_z(y_log[te], df["区县"].values[te], stats)
        sd_te = np.array([stats["per_district"].get(d, stats["global"])[1]
                          for d in df["区县"].values[te]])
        model.fit(X.iloc[tr], ytr_z)
        pred = model.predict(X.iloc[te])
        oof[te] = pred
        oof_yz[te] = yte_z
        oof_sd[te] = sd_te
        per_fold.append({
            "fold": k,
            "n_train": int(len(tr)),
            "n_test": int(len(te)),
            "r2": round(float(r2_score(yte_z, pred)), 4),
            "rmse": round(float(np.sqrt(mean_squared_error(yte_z, pred))), 4),
            "mae": round(float(mean_absolute_error(yte_z, pred)), 4),
        })
    w = np.array([f["n_test"] for f in per_fold], dtype=float)
    out = {
        "scheme": scheme_name,
        "model": model_name,
        "n": int(len(df)),
        "r2": round(float(np.average([f["r2"] for f in per_fold], weights=w)), 4),
        "rmse": round(float(np.average([f["rmse"] for f in per_fold], weights=w)), 4),
        "mae": round(float(np.average([f["mae"] for f in per_fold], weights=w)), 4),
        "r2_min_fold": round(min(f["r2"] for f in per_fold), 4),
        "r2_max_fold": round(max(f["r2"] for f in per_fold), 4),
        "metric_note": "按测试集样本数加权的折均值；不拼接各折 OOF 算全局 R²（各折 z 基准不同）",
        "per_fold": per_fold,
    }
    return out, {"pred": oof, "y_z": oof_yz, "sd": oof_sd}


def morans_i(values: np.ndarray, coords_km: np.ndarray, k: int = 8, n_perm: int = 999) -> dict:
    """8 邻域 KNN + 行标准化权重的 Moran's I，附置换检验。"""
    n = len(values)
    _, idx = cKDTree(coords_km).query(coords_km, k=k + 1)
    W = np.zeros((n, n), dtype=float)
    for i in range(n):
        W[i, idx[i, 1:]] = 1.0
    W = W / W.sum(axis=1, keepdims=True)
    z = values - values.mean()
    S0 = W.sum()
    I = float((n / S0) * (z @ W @ z) / (z @ z))
    rng = np.random.default_rng(RANDOM)
    perm = np.empty(n_perm)
    for r in range(n_perm):
        zp = rng.permutation(z)
        perm[r] = (n / S0) * (zp @ W @ zp) / (zp @ zp)
    p = float((np.sum(np.abs(perm) >= abs(I)) + 1) / (n_perm + 1))
    return {"I": round(I, 4), "p_perm": round(p, 4), "k_neighbors": k,
            "weight": "row-standardized KNN", "n": int(n),
            "expected_I": round(-1 / (n - 1), 4), "n_perm": n_perm}


def main() -> None:
    df = pd.read_csv(BUILD / "features.csv", encoding="utf-8-sig")
    flog = json.loads((BUILD / "feature_log.json").read_text(encoding="utf-8"))
    feats = flog["features"]
    X = df[feats].astype(float)
    y_log = np.log(df["price_per_sqm"].astype(float).values)
    district = df["区县"].values

    splits = scheme_splits(df)
    results, oof_store = [], {}
    for scheme_name, sp in splits.items():
        for model_name in ["dummy", "linear", "hgb"]:
            res, oof = run_cv(df, X, y_log, model_name, build_models()[model_name],
                              scheme_name, sp)
            results.append(res)
            oof_store[f"{scheme_name}|{model_name}"] = oof
            print(f"{scheme_name:14s} {model_name:7s} R²={res['r2']:+.3f} "
                  f"RMSE={res['rmse']:.3f} MAE={res['mae']:.3f}")

    # ---- 对照：剔除 complete_score（与五个距离同源，会偷走线性模型的分） ----
    feats_nc = [f for f in feats if f != "complete_score"]
    res_nc, _ = run_cv(df, X[feats_nc], y_log, "hgb", build_models()["hgb"],
                       "group_district", splits["group_district"])
    print(f"对照（不含 complete_score）group_district hgb R²={res_nc['r2']:+.3f}")

    # ---- 特征组消融：空间类（可达性+密度） vs 属性类（形态+配套） ------------
    # 这一组回答的是「空间到底解释了多少」——本项目最该被客户问的一句话
    man = pd.read_csv(BUILD / "feature_manifest.csv", encoding="utf-8-sig")
    grp = dict(zip(man["feature"], man["group"]))
    f_spatial = [f for f in feats if grp.get(f, "").startswith(("A", "B"))]
    f_attr = [f for f in feats if grp.get(f, "").startswith(("C", "D"))]
    abl = {}
    for tag, cols in [("spatial_only", f_spatial), ("attribute_only", f_attr),
                      ("all", feats)]:
        for scheme in ["group_district", "random"]:
            r, _ = run_cv(df, X[cols], y_log, "hgb", build_models()["hgb"], scheme,
                          splits[scheme])
            abl.setdefault(tag, {})[scheme] = {"r2": r["r2"], "n_features": len(cols)}
            print(f"消融 {tag:14s} ({len(cols):2d} 特征) {scheme:14s} R²={r['r2']:+.3f}")

    # ---- 置换重要性：在按区县切分的测试折上做，逐折取均值 --------------------
    imp = {f: [] for f in feats}
    for tr, te in splits["group_district"]:
        stats = fit_stats(y_log[tr], district[tr])
        ytr_z = apply_z(y_log[tr], district[tr], stats)
        yte_z = apply_z(y_log[te], district[te], stats)
        m = build_models()["hgb"].fit(X.iloc[tr], ytr_z)
        r = permutation_importance(m, X.iloc[te], yte_z, n_repeats=15,
                                   random_state=RANDOM, scoring="r2")
        for name, v in zip(feats, r.importances_mean):
            imp[name].append(float(v))
    imp_avg = {k: {"mean": round(float(np.mean(v)), 4),
                   "sd_across_folds": round(float(np.std(v)), 4)}
               for k, v in imp.items()}
    imp_avg = dict(sorted(imp_avg.items(), key=lambda kv: -abs(kv[1]["mean"])))

    # ---- 残差诊断（用按区县的 OOF，最严的一套） -----------------------------
    o = oof_store["group_district|hgb"]
    resid = o["y_z"] - o["pred"]
    coords = np.column_stack([
        df["lat"].values * 110.570,
        df["lon"].values * 111.320 * np.cos(np.radians(df["lat"].values.mean())),
    ])
    ok = ~np.isnan(resid)
    moran = morans_i(resid[ok], coords[ok])

    oof_df = df[["key", "小区名称", "区县", "片区", "lat", "lon", "price_per_sqm"]].copy()
    oof_df["y_z"] = o["y_z"]
    oof_df["pred_z"] = o["pred"]
    oof_df["resid_z"] = resid
    # 把 z 残差按该折该区县的 log 标准差还原成「预测单价相对实际的偏差百分比」
    oof_df["resid_pct"] = (np.exp(resid * o["sd"]) - 1) * 100
    oof_df.to_csv(BUILD / "oof_predictions.csv", index=False, encoding="utf-8-sig")

    # ---- 关键差值 -----------------------------------------------------------
    def r2(scheme, model):
        return next(r["r2"] for r in results
                    if r["scheme"] == scheme and r["model"] == model)

    gap = {
        m: {
            "r2_random": r2("random", m),
            "r2_group_area": r2("group_area", m),
            "r2_group_district": r2("group_district", m),
            "inflation_vs_area": round(r2("random", m) - r2("group_area", m), 4),
            "inflation_vs_district": round(r2("random", m) - r2("group_district", m), 4),
        }
        for m in ["linear", "hgb"]
    }

    payload = {
        "n_samples": int(len(df)),
        "n_features": len(feats),
        "target": "区县内 log(单价) 的 z-score（折内估计均值/标准差）",
        "results": results,
        "ablation_no_complete_score": res_nc,
        "ablation_feature_groups": abl,
        "cv_gap": gap,
        "importance": imp_avg,
        "moran_residual": moran,
        "random_state": RANDOM,
    }
    (BUILD / "cv_results.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\nCV 差值：", json.dumps(gap, ensure_ascii=False))
    print("残差 Moran's I：", moran)
    print("Top5 重要性：", list(imp_avg.items())[:5])
    print("\n残差按区县均值（百分点）：",
          df.assign(r=oof_df["resid_pct"].values).groupby("区县")["r"].mean().round(1).to_dict())


if __name__ == "__main__":
    main()
