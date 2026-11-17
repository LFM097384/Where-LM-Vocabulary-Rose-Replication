"""Length-adjusted marker outcome: expected marker types given prose-token count, from the pooled 2021-22 answers in fine ntok bins.
Outcome extadj = 100*(ext_types - E[ext_types|ntok bin]) / max(ntok,20); anyadj = any - E[any|bin]. Univariate and joint dose models, RI."""
import glob, numpy as np, pandas as pd, statsmodels.formula.api as smf
H = str(__import__("pathlib").Path(__file__).resolve().parents[1])
E = pd.read_csv(H + r"/results_all/exposure.csv"); E["rank"] = E.code_q.rank(pct=True) - .5
for c, src in [("r_size", "n_pre"), ("r_len", "lw_pre"), ("r_math", "math_a")]: E[c] = E[src].rank(pct=True) - .5
M = pd.concat([pd.read_parquet(f, columns=["id", "date", "ntok", "ext_types", "site"]) for f in glob.glob(H + r"/vocab/mk_*.parquet")]); M = M[M.site.isin(E.site)]
M = M[M.date.between("2019-01-01", "2024-04-01")]
edges = np.unique(np.r_[np.arange(0, 200, 5), np.arange(200, 1000, 25), np.arange(1000, 5000, 250), 1e9]); M["bin"] = np.digitize(M.ntok, edges)
M["any"] = (M.ext_types > 0).astype(float)
ref = M[(M.date >= "2021-01-01") & (M.date < "2022-12-01")].groupby("bin").agg(et=("ext_types", "mean"), ea=("any", "mean"))
M = M.join(ref, on="bin"); M["extadj"] = 100 * (M.ext_types - M.et) / M.ntok.clip(lower=20); M["anyadj"] = M["any"] - M.ea
M["month"] = M.date.str[:7]
P = M.groupby(["site", "month"]).agg(n=("id", "size"), extadj=("extadj", "mean"), anyadj=("anyadj", "mean"), lntok=("ntok", lambda v: np.log1p(v).mean())).reset_index()
P = P[P.month.between("2019-01", "2024-03")]; P["t"] = P.month.map(lambda m: (int(m[:4]) - 2022) * 12 + int(m[5:7]) - 12); P = P.merge(E, on="site"); P["post"] = (P.t >= 0).astype(int)
P.to_parquet(H + r"/results_vocab/panel_lenadj.parquet")
out = []
def fit(z, y, rhs, tr, pl):
    z = z.copy()
    if pl: z = z[z.t < 0]; z["post"] = (z.t >= -12).astype(int)
    m = smf.wls(f"{y} ~ C(site) + C(month) + {rhs}" + (" + C(site):t" if tr else ""), z, weights=np.minimum(z.n, 2000)).fit(cov_type="cluster", cov_kwds={"groups": z.site.astype("category").cat.codes})
    return m
for y in ["extadj", "anyadj", "lntok"]:
    for tr in [True, False]:
        for pl in [False, True]:
            for d in ["rank", "r_size", "r_len", "r_math"]:
                m = fit(P, y, f"{d}:post", tr, pl); out.append(dict(outcome=y, trend=tr, placebo=pl, model="single", term=d, b=m.params[d + ":post"], p=m.pvalues[d + ":post"]))
            m = fit(P, y, "rank:post + r_size:post + r_len:post + r_math:post", tr, pl)
            for d in ["rank", "r_size", "r_len", "r_math"]: out.append(dict(outcome=y, trend=tr, placebo=pl, model="joint", term=d, b=m.params[d + ":post"], p=m.pvalues[d + ":post"]))
O = pd.DataFrame(out); O.to_csv(H + r"/results_vocab/lenadj_doses.csv", index=False); pd.set_option("display.width", 200); pd.set_option("display.max_rows", 300)
print(O[~O.placebo].round(4).to_string()); print(O[O.placebo].round(4).to_string())
