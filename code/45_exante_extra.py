"""Extra ex-ante checks: joint code+edit panel without interaction; 2019-20 answers edited within 30 days as an alternative pre-period
measure; site-level screen with it; self/other 30-day split in panel already in exante_models.csv."""
import glob, numpy as np, pandas as pd, statsmodels.formula.api as smf
H = str(__import__("pathlib").Path(__file__).resolve().parents[1]); R = H + r"/results_vocab"
S = pd.read_csv(R + r"/exante_edit.csv").set_index("site")
X = pd.concat([pd.read_parquet(f, columns=["date", "first_self", "first_other", "site"]) for f in glob.glob(H + r"/data_edit/edits_*.parquet")])
X = X[(X.date >= "2019-01") & (X.date < "2021-01")]; d0 = pd.to_datetime(X.date.str[:19])
f = pd.concat([pd.to_datetime(X.first_self.str[:19], errors="coerce"), pd.to_datetime(X.first_other.str[:19], errors="coerce")], axis=1).min(axis=1)
X["e30"] = (f - d0).dt.days.le(30); e19 = X.groupby("site").e30.mean(); S["r_e19"] = e19.reindex(S.index).rank(pct=True) - .5
print("spearman ea30 vs e19", round(S.ea30.corr(e19.reindex(S.index), method="spearman"), 3))
L = pd.read_csv(R + r"/site_logchange2.csv").set_index("site")[["y", "w", "rank"]].join(S[["r_ea30", "r_e19"]], how="inner")
for fm in ["r_e19", "rank + r_e19"]:
    m = smf.wls("y ~ " + fm, L, weights=L.w).fit(cov_type="HC1"); print("screen", fm, m.params.round(3).to_dict(), m.pvalues.round(4).to_dict(), round(m.rsquared, 3))
P = pd.read_parquet(R + r"/panel_edit.parquet").merge(S[["r_ea30", "r_e19"]].reset_index(), on="site")
for y in ["ext", "anyx"]:
    for tr in [False, True]:
        for pl in [False, True]:
            z = P.dropna(subset=[y, "r_e19"]).copy()
            if pl: z = z[z.t < 0]; z["post"] = (z.t >= -12).astype(int)
            for rhs in ["rank:post + r_ea30:post", "r_e19:post"]:
                m = smf.wls(f"{y} ~ C(site) + C(month) + {rhs}" + (" + C(site):t" if tr else ""), z, weights=np.minimum(z.n, 2000)).fit(cov_type="cluster", cov_kwds={"groups": z.site.astype("category").cat.codes})
                print(y, "trend", tr, "placebo", pl, rhs, {k: (round(m.params[k], 4), round(m.pvalues[k], 4)) for k in m.params.index if k.endswith(":post")})

