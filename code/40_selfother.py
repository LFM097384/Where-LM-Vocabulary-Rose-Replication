"""Split the edit share into self-edits and edits by others (2021-01..2022-11 answers).
Edits counted only if made within 30 days of posting (removes the censoring asymmetry between old and new answers) and also any time.
Site screen (same statistic as 38_screen2: log change marker vs matched control, 2021Q1-22Q3 -> 2023Q2-24Q1), monthly panel with
site + month FE, placebo windows. Also: answer-level marker split by self/other-edited among post answers."""
import os, glob, numpy as np, pandas as pd, statsmodels.formula.api as smf
H = str(__import__("pathlib").Path(__file__).resolve().parents[1]); R = H + r"/results_vocab"; DE = H + r"/data_edit"
fs = glob.glob(DE + r"/edits_*.parquet"); X = pd.concat([pd.read_parquet(f) for f in fs])
X["d"] = pd.to_datetime(X.date.str[:19]); 
for c in ["first_self", "first_other"]: X[c + "_d"] = pd.to_datetime(X[c].str[:19], errors="coerce")
X["self30"] = (X.first_self_d - X.d).dt.days.le(30); X["other30"] = (X.first_other_d - X.d).dt.days.le(30)
X["self_any"] = X.n_self > 0; X["other_any"] = X.n_other > 0; X["any"] = X.self_any | X.other_any
pre = X[(X.date >= "2021-01") & (X.date < "2022-12")]
S = pre.groupby("site")[["self_any", "other_any", "any", "self30", "other30"]].mean(); S["n_ans"] = pre.groupby("site").size()
print("sites with edit data:", len(S)); print(S.describe().round(3))
print("corr:", S.corr(method="spearman").round(2))
# drift of other-edit share by year (censoring check)
X["yr"] = X.date.str[:4]; print(X.groupby("yr")[["self_any", "other_any", "self30", "other30"]].mean().round(3))
S.to_csv(R + r"/edit_selfother.csv")
for c in ["self_any", "other_any", "any", "self30", "other30"]: S["r_" + c] = S[c].rank(pct=True) - .5
L = pd.read_csv(R + r"/site_logchange2.csv").set_index("site").join(S, how="inner")
rows = []
def fit(f, d, lab):
    m = smf.wls("y ~ " + f, d, weights=d.w).fit(cov_type="HC1")
    for k in f.split(" + "): rows.append(dict(model=lab, spec=f, term=k, b=m.params[k], p=m.pvalues[k], r2=m.rsquared, sites=len(d)))
for f in ["r_edit", "r_any", "r_self_any", "r_other_any", "r_self30", "r_other30", "r_self_any + r_other_any", "r_self30 + r_other30",
          "rank + r_self_any + r_other_any", "rank + r_self30 + r_other30"]:
    fit(f, L, "screen_chatgpt")
# placebo screen window using ddd_panel
D = pd.read_csv(R + r"/ddd_panel.csv")
plA = ["2019Q1", "2019Q2", "2019Q3", "2019Q4", "2020Q1", "2020Q2", "2020Q3"]; plB = ["2021Q1", "2021Q2", "2021Q3", "2021Q4"]
g = D[D.q.isin(plA)].groupby("site").agg(mk=("mk", "mean"), ctl=("ctl", "mean"), n=("n", "sum")); h = D[D.q.isin(plB)].groupby("site").agg(mk=("mk", "mean"), ctl=("ctl", "mean"), n=("n", "sum"))
Pl = pd.DataFrame({"y": np.log(h.mk / g.mk) - np.log(h.ctl / g.ctl), "w": np.sqrt(h.n + g.n)}).replace([np.inf, -np.inf], np.nan).dropna().join(S, how="inner")
for f in ["r_self_any", "r_other_any", "r_self_any + r_other_any", "r_self30 + r_other30"]: fit(f, Pl, "screen_placebo")
# monthly panel
P = pd.read_parquet(R + r"/panel_edit.parquet").merge(S[[c for c in S.columns if c.startswith("r_")]].reset_index(), on="site")
for tr in [False, True]:
    for pl in [False, True]:
        z = P.dropna(subset=["ext"]).copy()
        if pl: z = z[z.t < 0]; z["post"] = (z.t >= -12).astype(int)
        for terms in [["r_self_any", "r_other_any"], ["r_self30", "r_other30"], ["rank", "r_self_any", "r_other_any"]]:
            f = "ext ~ C(site) + C(month) + " + " + ".join(k + ":post" for k in terms) + (" + C(site):t" if tr else "")
            m = smf.wls(f, z, weights=np.minimum(z.n, 2000)).fit(cov_type="cluster", cov_kwds={"groups": z.site.astype("category").cat.codes})
            for k in terms: rows.append(dict(model=f"panel trend={tr} placebo={pl}", spec="+".join(terms), term=k, b=m.params[k + ":post"], p=m.pvalues[k + ":post"], r2=np.nan, sites=z.site.nunique()))
O = pd.DataFrame(rows); O.to_csv(R + r"/edit_selfother_models.csv", index=False)
pd.set_option("display.width", 220); pd.set_option("display.max_rows", 300); print(O.round(4).to_string())
# terciles of other-edit share
L["t_other"] = pd.qcut(L.other_any, 3, labels=["low", "mid", "high"]); L["t_self"] = pd.qcut(L.self_any, 3, labels=["low", "mid", "high"])
for c in ["t_other", "t_self"]: print(c, L.groupby(c).apply(lambda x: pd.Series({"y": np.average(x.y, weights=x.w), "n": len(x)})).round(3).to_dict())
