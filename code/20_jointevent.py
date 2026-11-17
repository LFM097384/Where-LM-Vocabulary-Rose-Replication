"""Joint half-year event study (no trends): ext and anyx (raw) on code rank and size rank together (and length rank), ref 2022H2.
Also randomization inference for joint no-trend code coefficient (permute code rank only, keeping size/len/math fixed)."""
import numpy as np, pandas as pd, statsmodels.formula.api as smf, glob
H = str(__import__("pathlib").Path(__file__).resolve().parents[1]); R = H + r"/results_vocab"
E = pd.read_csv(H + r"/results_all/exposure.csv"); E["rank"] = E.code_q.rank(pct=True) - .5
for c, src in [("r_size", "n_pre"), ("r_len", "lw_pre"), ("r_math", "math_a")]: E[c] = E[src].rank(pct=True) - .5
M = pd.concat([pd.read_parquet(f, columns=["id", "date", "ntok", "ext_types", "site"]) for f in glob.glob(H + r"/vocab/mk_*.parquet")]); M = M[M.site.isin(E.site)]
M["month"] = M.date.str[:7]; M["ext"] = 100 * M.ext_types / M.ntok.clip(lower=20); M["anyx"] = (M.ext_types > 0).astype(float)
P = M.groupby(["site", "month"]).agg(n=("id", "size"), ext=("ext", "mean"), anyx=("anyx", "mean")).reset_index()
P = P[P.month.between("2019-01", "2024-03")]; P["t"] = P.month.map(lambda m: (int(m[:4]) - 2022) * 12 + int(m[5:7]) - 12); P = P.merge(E, on="site"); P["post"] = (P.t >= 0).astype(int)
P["yr"] = P.month.str[:4] + np.where(P.month.str[5:7].astype(int) <= 6, "H1", "H2")
rows = []
for y in ["ext", "anyx"]:
    terms = []
    for pp in sorted(P.yr.unique()):
        if pp == "2022H2": continue
        for d in ["rank", "r_size", "r_len"]:
            c = f"{d}_{pp}"; P[c] = (P.yr == pp) * P[d]; terms.append(c)
    m = smf.wls(f"{y} ~ C(site) + C(month) + " + " + ".join(terms), P, weights=np.minimum(P.n, 2000)).fit(cov_type="cluster", cov_kwds={"groups": P.site.astype("category").cat.codes})
    for t in terms: d, pp = t.rsplit("_", 1); rows.append(dict(outcome=y, dose=d, yr=pp, b=m.params[t], se=m.bse[t]))
    for d in ["rank", "r_size", "r_len"]: rows.append(dict(outcome=y, dose=d, yr="2022H2", b=0.0, se=0.0))
EV = pd.DataFrame(rows).sort_values(["outcome", "dose", "yr"]); EV.to_csv(R + r"/event_joint.csv", index=False)
pd.set_option("display.width", 200); print(EV.pivot_table(index="yr", columns=["outcome", "dose"], values="b").round(4).to_string())
print(EV.pivot_table(index="yr", columns=["outcome", "dose"], values="se").round(4).to_string())
# RI on joint no-trend
z = P.reset_index(drop=True); w = np.sqrt(np.minimum(z.n, 2000).values)
S = pd.get_dummies(z.site).values.astype(float); Mo = pd.get_dummies(z.month).values[:, 1:].astype(float)
X = np.hstack([S, Mo, (z[["r_size", "r_len", "r_math"]].values * z.post.values[:, None])]) * w[:, None]; Q, _ = np.linalg.qr(X); res = lambda v: v - Q @ (Q.T @ v)
us = np.unique(z.site); pos = np.searchsorted(us, z.site.values); rk = z.drop_duplicates("site").set_index("site")["rank"].reindex(us).values
for y in ["ext", "anyx"]:
    yt = res(z[y].values * w)
    def beta(r): x = res(r[pos] * z.post.values * w); return (x @ yt) / (x @ x)
    b0 = beta(rk); rng = np.random.default_rng(11); bs = np.array([beta(rng.permutation(rk)) for _ in range(2000)])
    print("RI joint notrend", y, round(b0, 5), "p", (np.abs(bs) >= abs(b0)).mean())
