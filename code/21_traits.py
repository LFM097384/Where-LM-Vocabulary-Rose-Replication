"""Trait battery: pre-period (2021-01..2022-11) site traits as doses for the marker rate (and length-adjusted), univariate,
with and without site trends, plus placebo date. Shows where code exposure sits in the distribution of 'effects' of arbitrary traits."""
import numpy as np, pandas as pd, statsmodels.formula.api as smf
H = str(__import__("pathlib").Path(__file__).resolve().parents[1]); R = H + r"/results_vocab"
E = pd.read_csv(H + r"/results_all/exposure.csv")
pn = pd.read_csv(H + r"/results_all/panel.csv"); pp = pn[pn.month.between("2021-01", "2022-11")]
T = pp.groupby("site").apply(lambda g: pd.Series({c: np.average(g[c].fillna(g[c].mean()), weights=g.n.fillna(0) + 1e-9) for c in ["list", "head", "edited", "lw_sd", "lqw_m", "heres", "summary", "dv"]}))
T["lnq"] = pp.groupby("site").nq.mean().pipe(np.log1p); T["apq"] = pp.groupby("site").n.sum() / pp.groupby("site").nq.sum()
T = T.join(E.set_index("site")[["code_q", "code_a", "math_a", "n_pre", "lw_pre"]])
P = pd.read_parquet(R + r"/panel_lenadj.parquet")[["site", "month", "n", "extadj", "anyadj", "t", "post"]]
import glob
M = pd.concat([pd.read_parquet(f, columns=["id", "date", "ntok", "ext_types", "site"]) for f in glob.glob(H + r"/vocab/mk_*.parquet")])
M["month"] = M.date.str[:7]; M["ext"] = 100 * M.ext_types / M.ntok.clip(lower=20)
P = P.merge(M.groupby(["site", "month"]).ext.mean().reset_index(), on=["site", "month"])
traits = list(T.columns); Tr = T.rank(pct=True) - .5; Tr.columns = ["d_" + c for c in traits]
P = P.merge(Tr, left_on="site", right_index=True)
print(T.rank().corr("spearman")["code_q"].round(2).to_string())
out = []
for c in traits:
    for y in ["ext", "extadj"]:
        for tr in [True, False]:
            for pl in [False, True]:
                z = P.copy()
                if pl: z = z[z.t < 0]; z["post"] = (z.t >= -12).astype(int)
                m = smf.wls(f"{y} ~ C(site) + C(month) + d_{c}:post" + (" + C(site):t" if tr else ""), z, weights=np.minimum(z.n, 2000)).fit(cov_type="cluster", cov_kwds={"groups": z.site.astype("category").cat.codes})
                k = f"d_{c}:post"; out.append(dict(trait=c, outcome=y, trend=tr, placebo=pl, b=m.params[k], t=m.tvalues[k], p=m.pvalues[k]))
    print(c, flush=True)
O = pd.DataFrame(out); O.to_csv(R + r"/trait_battery.csv", index=False); pd.set_option("display.width", 200)
print(O.pivot_table(index="trait", columns=["outcome", "trend", "placebo"], values="t").round(2).to_string())
