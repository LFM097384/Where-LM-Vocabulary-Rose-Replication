"""Design-based checks requested by review.
(1) Randomization inference: permute exposure rank across sites (2000 draws), FWL with precomputed FE+trend projection.
(2) Placebo doses: pre-period size, pre-period mean log length, math share, all as centred ranks, same spec.
(3) Mechanics: marker rate among answers without list/header; prose tokens (log) as outcome; marker rate with ntok>=50.
(4) Composition: incumbents (first answer on site before 2022-12) only; post-period entrant-minus-incumbent gap on rank.
(5) Quarterly event study (marker rate), ref 2022Q4 = Oct-Nov 2022 (Dec 2022 starts post), to locate the 2023 peak vs strike (Jun-Aug 2023).
(6) Site-quarter word score dropping 'ensure', and leave-one-word-out range."""
import glob, os, numpy as np, pandas as pd, statsmodels.formula.api as smf
H = str(__import__("pathlib").Path(__file__).resolve().parents[1]); R = H + r"/results_vocab"
E = pd.read_csv(H + r"/results_all/exposure.csv"); E["rank"] = E.code_q.rank(pct=True) - .5
for c, src in [("r_size", "n_pre"), ("r_len", "lw_pre"), ("r_math", "math_a")]: E[c] = E[src].rank(pct=True) - .5
A = []
for s in E.site:
    f = H + rf"/vocab/mk_{s}.parquet"
    if not os.path.exists(f): continue
    m = pd.read_parquet(f, columns=["id", "owner", "date", "ntok", "ext_types"])
    a = pd.read_parquet(H + rf"/data_all/answers_{s}.parquet", columns=["id", "owner", "date", "nli", "nhd"])
    first = a[a.owner > 0].groupby("owner").date.min()
    m = m.merge(a[["id", "nli", "nhd"]], on="id", how="left"); m["first"] = m.owner.map(first); m["site"] = s; A.append(m)
A = pd.concat(A); A = A[A.date.between("2019-01-01", "2024-04-01")]
A["month"] = A.date.str[:7]; A["ext"] = 100 * A.ext_types / A.ntok.clip(lower=20); A["anyx"] = (A.ext_types > 0).astype(float)
A["inc"] = A["first"].notna() & (A["first"] < "2022-12-01"); A["plain"] = (A.nli.fillna(0) == 0) & (A.nhd.fillna(0) == 0)
A["lntok"] = np.log1p(A.ntok)
def panel(d, cols=("ext", "anyx", "lntok")):
    g = d.groupby(["site", "month"]).agg(n=("id", "size"), **{c: (c, "mean") for c in cols}).reset_index()
    g["t"] = g.month.map(lambda m: (int(m[:4]) - 2022) * 12 + int(m[5:7]) - 12); g = g.merge(E, on="site"); g["post"] = (g.t >= 0).astype(int); return g
def fit(z, y="ext", dose="rank", trend=True, placebo=False):
    z = z.dropna(subset=[y]).copy()
    if placebo: z = z[z.t < 0]; z["post"] = (z.t >= -12).astype(int)
    m = smf.wls(f"{y} ~ C(site) + C(month) + {dose}:post" + (" + C(site):t" if trend else ""), z, weights=np.minimum(z.n, 2000)).fit(
        cov_type="cluster", cov_kwds={"groups": z.site.astype("category").cat.codes})
    k = f"{dose}:post"; return m.params[k], m.bse[k], m.pvalues[k]
P = panel(A); out = []
# (1) randomization inference via FWL
def ri(z, y, trend, draws=2000, seed=7):
    z = z.dropna(subset=[y]).reset_index(drop=True); w = np.sqrt(np.minimum(z.n, 2000).values)
    S = pd.get_dummies(z.site).values.astype(float); Mo = pd.get_dummies(z.month).values[:, 1:].astype(float)
    X = np.hstack([S, Mo] + ([S * z.t.values[:, None]] if trend else [])) * w[:, None]
    Q, _ = np.linalg.qr(X); res = lambda v: v - Q @ (Q.T @ v)
    yt = res(z[y].values * w); sites = z.site.values; us = np.unique(sites); rk = z.drop_duplicates("site").set_index("site")["rank"].reindex(us).values
    pos = np.searchsorted(us, sites); post = z.post.values
    def beta(r): x = res(r[pos] * post * w); return (x @ yt) / (x @ x)
    b0 = beta(rk); rng = np.random.default_rng(seed); bs = np.array([beta(rng.permutation(rk)) for _ in range(draws)])
    return b0, (np.abs(bs) >= abs(b0)).mean(), np.quantile(np.abs(bs), .95)
for y in ["ext", "anyx"]:
    for tr in [True, False]:
        b0, p, q95 = ri(P, y, tr); out.append(dict(check="RI site permutation", outcome=y, trend=tr, b=b0, p=p, note=f"|b| q95 null={q95:.4f}"))
        print(out[-1], flush=True)
# (2) placebo doses
for d in ["r_size", "r_len", "r_math"]:
    for y in ["ext", "anyx"]:
        b, se, p = fit(P, y, d); out.append(dict(check=f"placebo dose {d}", outcome=y, trend=True, b=b, p=p))
# (3) mechanics
Pp = panel(A[A.plain]); b, se, p = fit(Pp); out.append(dict(check="answers without list/header", outcome="ext", trend=True, b=b, p=p))
b, se, p = fit(P, "lntok"); out.append(dict(check="prose tokens outcome", outcome="lntok", trend=True, b=b, p=p))
b, se, p = fit(P, "lntok", placebo=True); out.append(dict(check="prose tokens outcome placebo", outcome="lntok", trend=True, b=b, p=p))
Pl = panel(A[A.ntok >= 50]); b, se, p = fit(Pl); out.append(dict(check="answers with >=50 prose tokens", outcome="ext", trend=True, b=b, p=p))
# (4) composition
Pi = panel(A[A.inc]); 
for tr in [True, False]:
    b, se, p = fit(Pi, trend=tr); out.append(dict(check="incumbents only", outcome="ext", trend=tr, b=b, p=p))
    b, se, p = fit(Pi, "anyx", trend=tr); out.append(dict(check="incumbents only", outcome="anyx", trend=tr, b=b, p=p))
post = A[A.date >= "2022-12-01"]; g = post.groupby(["site", "inc"]).ext.mean().unstack().dropna(); g["gap"] = g[False] - g[True]
nn = post.groupby("site").size(); g = g.join(E.set_index("site")["rank"]); g["n"] = nn.reindex(g.index)
m = smf.wls("gap ~ rank", g, weights=np.minimum(g.n, 5000)).fit(cov_type="HC1"); out.append(dict(check="entrant minus incumbent gap (post) on rank", outcome="ext", trend=False, b=m.params["rank"], p=m.pvalues["rank"], note=f"mean gap={g.gap.mean():.4f}"))
share = post.groupby("site").inc.mean().rename("inc_share").to_frame().join(E.set_index("site")["rank"])
m = smf.ols("inc_share ~ rank", share).fit(cov_type="HC1"); out.append(dict(check="post incumbent share on rank", outcome="inc_share", trend=False, b=m.params["rank"], p=m.pvalues["rank"], note=f"mean={share.inc_share.mean():.3f}"))
# (5) quarterly event study
P["qq"] = P.month.map(lambda m: m[:4] + "Q" + str((int(m[5:7]) - 1) // 3 + 1)); P.loc[P.month == "2022-12", "qq"] = "2022Q4b"
terms = []
for q in sorted(P.qq.unique()):
    if q < "2021Q1" or q == "2022Q4": continue
    c = "e_" + q; P[c] = (P.qq == q) * P["rank"]; terms.append(c)
m = smf.wls("ext ~ C(site) + C(month) + " + " + ".join(terms), P, weights=np.minimum(P.n, 2000)).fit(cov_type="cluster", cov_kwds={"groups": P.site.astype("category").cat.codes})
EV = pd.DataFrame({"q": [t[2:] for t in terms], "b": [m.params[t] for t in terms], "se": [m.bse[t] for t in terms]}); EV.to_csv(R + r"/event_quarter_marker.csv", index=False)
O = pd.DataFrame(out); O.to_csv(R + r"/design_checks.csv", index=False)
pd.set_option("display.width", 220); print(O.round(4).to_string()); print(EV.round(4).to_string())
