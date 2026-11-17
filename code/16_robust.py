"""Robustness for the marker rate: (a) drop top-10 exposure sites; (b) drop the 10 largest sites; (c) binary top vs bottom tercile;
(d) unweighted; (e) wild-ish check: site-level DiD (post minus pre mean, detrended) regressed on rank -> scatter figure; (f) linear raw share."""
import glob, numpy as np, pandas as pd, statsmodels.formula.api as smf, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
H = str(__import__("pathlib").Path(__file__).resolve().parents[1])
E = pd.read_csv(H + r"/results_all/exposure.csv"); E["rank"] = E.code_q.rank(pct=True) - .5
M = pd.concat([pd.read_parquet(f, columns=["id", "date", "ntok", "ext_types", "site"]) for f in glob.glob(H + r"/vocab/mk_*.parquet")]); M = M[M.site.isin(E.site)]
M["month"] = M.date.str[:7]; M["ext"] = 100 * M.ext_types / M.ntok.clip(lower=20); M["anyx"] = (M.ext_types > 0).astype(float)
P = M.groupby(["site", "month"]).agg(n=("id", "size"), ext=("ext", "mean"), anyx=("anyx", "mean")).reset_index()
P = P[P.month.between("2019-01", "2024-03")]; P["t"] = P.month.map(lambda m: (int(m[:4]) - 2022) * 12 + int(m[5:7]) - 12)
P = P.merge(E[["site", "rank", "code_q"]], on="site"); P["post"] = (P.t >= 0).astype(int)
def fit(z, f="rank:post", w=True, trend=True, y="ext"):
    m = smf.wls(f"{y} ~ C(site) + C(month) + {f}" + (" + C(site):t" if trend else ""), z, weights=np.minimum(z.n, 2000) if w else np.ones(len(z))).fit(
        cov_type="cluster", cov_kwds={"groups": z.site.astype("category").cat.codes})
    k = f.split(" ")[0]; return m.params[k], m.bse[k], m.pvalues[k], z.site.nunique()
top10 = E.nlargest(10, "code_q").site; big10 = P.groupby("site").n.sum().nlargest(10).index
res = []
res.append(("baseline",) + fit(P))
res.append(("drop 10 most exposed",) + fit(P[~P.site.isin(top10)]))
res.append(("drop 10 largest",) + fit(P[~P.site.isin(big10)]))
res.append(("unweighted",) + fit(P, w=False))
res.append(("raw code share",) + fit(P, "code_q:post"))
T = P.copy(); T["ter"] = pd.qcut(T["rank"], 3, labels=[0, 1, 2]).astype(int); T = T[T.ter != 1]; T["top"] = (T.ter == 2).astype(int)
res.append(("top vs bottom tercile",) + fit(T, "top:post"))
res.append(("2021 on, no trend",) + fit(P[P.month >= "2021-01"], trend=False))
R_ = pd.DataFrame(res, columns=["spec", "b", "se", "p", "sites"]); R_.to_csv(H + r"/results_vocab/robust_marker.csv", index=False); print(R_.round(4).to_string())
# site-level detrended change: residualize month FE, then per site fit pre trend on 2019-01..2022-11 and take mean post residual 2023-01..2023-12
P["res"] = P.ext - P.groupby("month").ext.transform(lambda v: np.average(v, weights=np.minimum(P.loc[v.index, "n"], 2000)))
rows = []
for s, g in P.groupby("site"):
    pre = g[g.t < 0]; post = g[(g.month >= "2023-01") & (g.month <= "2023-12")]
    if len(pre) < 24 or len(post) < 6: continue
    b = np.polyfit(pre.t, pre.res, 1, w=np.sqrt(np.minimum(pre.n, 2000))); d = (post.res - np.polyval(b, post.t)).mean()
    rows.append(dict(site=s, rank=g["rank"].iloc[0], d=d, n=g.n.sum()))
S = pd.DataFrame(rows); S.to_csv(H + r"/results_vocab/site_change.csv", index=False)
fig, ax = plt.subplots(figsize=(3.3, 2.3)); plt.rcParams.update({"font.family": "serif"})
ax.scatter(S["rank"] + .5, S.d, s=np.clip(S.n / 4000, 2, 40), color="k", alpha=.45, lw=0)
bins = pd.cut(S["rank"], 10); g = S.groupby(bins, observed=True).agg(x=("rank", "mean"), y=("d", "median"))
ax.plot(g.x + .5, g.y, color="#b2182b", lw=1.2); ax.axhline(0, color="grey", lw=.5)
for s in ["codereview", "tex", "unix", "salesforce", "coffee", "parenting", "history", "math"]:
    r = S[S.site == s]
    if len(r): ax.annotate(s, (r["rank"].iloc[0] + .5, r.d.iloc[0]), fontsize=5, xytext=(2, 2), textcoords="offset points")
ax.set_ylim(S.d.quantile(.01) * 1.3, S.d.quantile(.99) * 1.3); ax.set_xlabel("code-exposure percentile", fontsize=7); ax.set_ylabel("2023 marker rate vs. own trend", fontsize=7)
ax.tick_params(labelsize=6); ax.spines[["top", "right"]].set_visible(False); plt.tight_layout()
plt.savefig(H + r"/paper/fig_sites.pdf"); plt.savefig(H + r"/paper/fig_sites.png", dpi=160)
print(g.round(4).to_string()); print("spearman", S[["rank", "d"]].corr("spearman").iloc[0, 1])
