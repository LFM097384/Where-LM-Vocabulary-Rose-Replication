"""Descriptives and remaining checks for the ex-ante (30-day) edit share."""
import glob, numpy as np, pandas as pd, statsmodels.formula.api as smf
H = str(__import__("pathlib").Path(__file__).resolve().parents[1]); R = H + r"/results_vocab"
S = pd.read_csv(R + r"/exante_edit.csv").set_index("site"); E = pd.read_csv(R + r"/exposure_plus.csv").set_index("site")
print("ea30 quantiles", S.ea30.quantile([.25, .5, .75]).round(3).to_dict(), "self", S.ea30_self.median().round(3), "other", S.ea30_other.median().round(3))
print("spearman ea30 code", round(S.ea30.corr(E.code_q.reindex(S.index), method="spearman"), 3), "self-other", round(S.ea30_self.corr(S.ea30_other, method="spearman"), 3))
X = pd.concat([pd.read_parquet(f) for f in glob.glob(H + r"/data_edit/edits_*.parquet")]); d0 = pd.to_datetime(X.date.str[:19])
fs = pd.to_datetime(X.first_self.str[:19], errors="coerce"); fo = pd.to_datetime(X.first_other.str[:19], errors="coerce")
X["e30"] = (pd.concat([fs, fo], axis=1).min(axis=1) - d0).dt.days.le(30); X["s30"] = (fs - d0).dt.days.le(30); X["o30"] = (fo - d0).dt.days.le(30)
X["yr"] = X.date.str[:4]; X = X[X.date < "2024-03"]
ter = pd.qcut(S.ea30.rank(method="first"), 3, labels=["low", "mid", "high"]); X["ter"] = X.site.map(ter)
Y = X.groupby(["site", "yr"])[["e30", "s30", "o30"]].mean().reset_index(); Y["ter"] = Y.site.map(ter)
print(Y.groupby(["ter", "yr"]).e30.median().unstack().round(3).to_string()); print(Y.groupby("yr")[["s30", "o30"]].median().round(3).to_string())
L = pd.read_csv(R + r"/site_logchange2.csv").set_index("site")[["y", "w", "rank"]].join(S, how="inner")
for f in ["r_ea30_self", "r_ea30_other"]:
    m = smf.wls("y ~ " + f, L, weights=L.w).fit(cov_type="HC1"); print("screen", f, round(m.params[f], 3), round(m.pvalues[f], 4), round(m.rsquared, 3))
rng = np.random.default_rng(7)
for c in ["ea30_self", "ea30_other"]:
    t = pd.qcut(L[c].rank(method="first"), 3, labels=[0, 1, 2]).astype(int); print("terciles", c, [round(np.average(L.y[t == k], weights=L.w[t == k]), 3) for k in range(3)])
