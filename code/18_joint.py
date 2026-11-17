"""Joint model with code rank + size, length, math ranks; correlations; plus code effect within size/length terciles."""
import glob, numpy as np, pandas as pd, statsmodels.formula.api as smf
H = str(__import__("pathlib").Path(__file__).resolve().parents[1])
E = pd.read_csv(H + r"/results_all/exposure.csv"); E["rank"] = E.code_q.rank(pct=True) - .5
for c, src in [("r_size", "n_pre"), ("r_len", "lw_pre"), ("r_math", "math_a")]: E[c] = E[src].rank(pct=True) - .5
print(E[["rank", "r_size", "r_len", "r_math"]].corr("spearman").round(2))
M = pd.concat([pd.read_parquet(f, columns=["id", "date", "ntok", "ext_types", "site"]) for f in glob.glob(H + r"/vocab/mk_*.parquet")]); M = M[M.site.isin(E.site)]
M["month"] = M.date.str[:7]; M["ext"] = 100 * M.ext_types / M.ntok.clip(lower=20); M["anyx"] = (M.ext_types > 0).astype(float)
P = M.groupby(["site", "month"]).agg(n=("id", "size"), ext=("ext", "mean"), anyx=("anyx", "mean")).reset_index()
L = pd.read_csv(H + r"/results_all/panel.csv")[["site", "month", "list"]]; P = P.merge(L, on=["site", "month"], how="left")
P = P[P.month.between("2019-01", "2024-03")]; P["t"] = P.month.map(lambda m: (int(m[:4]) - 2022) * 12 + int(m[5:7]) - 12); P = P.merge(E, on="site"); P["post"] = (P.t >= 0).astype(int)
out = []
for y in ["ext", "anyx", "list"]:
    for tr in [True, False]:
        for pl in [False, True]:
            z = P.dropna(subset=[y]).copy()
            if pl: z = z[z.t < 0]; z["post"] = (z.t >= -12).astype(int)
            m = smf.wls(f"{y} ~ C(site) + C(month) + rank:post + r_size:post + r_len:post + r_math:post" + (" + C(site):t" if tr else ""), z, weights=np.minimum(z.n, 2000)).fit(cov_type="cluster", cov_kwds={"groups": z.site.astype("category").cat.codes})
            for k in ["rank", "r_size", "r_len", "r_math"]: out.append(dict(outcome=y, trend=tr, placebo=pl, term=k, b=m.params[k + ":post"], se=m.bse[k + ":post"], p=m.pvalues[k + ":post"]))
O = pd.DataFrame(out); O.to_csv(H + r"/results_vocab/joint_doses.csv", index=False); pd.set_option("display.width", 200); print(O.round(4).to_string())
