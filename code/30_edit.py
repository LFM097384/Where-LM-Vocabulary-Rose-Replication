"""Edit-culture dose. r_edit = centred rank of pre-period (2021-01..2022-11) share of answers ever edited.
(a) monthly panel ext/anyx/list/lw: single r_edit, joint with code rank (+size,len,math), trends, placebo, RI.
(b) half-year event study joint code+edit. (c) leave-crypto-out (sites with blockchain topics). (d) top contributors."""
import glob, numpy as np, pandas as pd, statsmodels.formula.api as smf
H = str(__import__("pathlib").Path(__file__).resolve().parents[1]); R = H + r"/results_vocab"
E = pd.read_csv(H + r"/results_all/exposure.csv")
pn = pd.read_csv(H + r"/results_all/panel.csv"); pp = pn[pn.month.between("2021-01", "2022-11")]
ed = pp.groupby("site").apply(lambda g: np.average(g.edited.fillna(0), weights=g.n.fillna(0) + 1e-9)).rename("edited")
E = E.merge(ed.reset_index(), on="site"); E["rank"] = E.code_q.rank(pct=True) - .5
for c, s_ in [("r_size", "n_pre"), ("r_len", "lw_pre"), ("r_math", "math_a"), ("r_edit", "edited")]: E[c] = E[s_].rank(pct=True) - .5
E.to_csv(R + r"/exposure_plus.csv", index=False)
M = pd.concat([pd.read_parquet(f, columns=["id", "date", "ntok", "ext_types", "site"]) for f in glob.glob(H + r"/vocab/mk_*.parquet")]); M = M[M.site.isin(E.site)]
M["month"] = M.date.str[:7]; M["ext"] = 100 * M.ext_types / M.ntok.clip(lower=20); M["anyx"] = (M.ext_types > 0).astype(float)
P = M.groupby(["site", "month"]).agg(n=("id", "size"), ext=("ext", "mean"), anyx=("anyx", "mean")).reset_index().merge(pn[["site", "month", "list", "head", "lw"]], on=["site", "month"], how="left")
P = P[P.month.between("2019-01", "2024-03")]; P["t"] = P.month.map(lambda m: (int(m[:4]) - 2022) * 12 + int(m[5:7]) - 12); P = P.merge(E, on="site"); P["post"] = (P.t >= 0).astype(int)
P.to_parquet(R + r"/panel_edit.parquet")
def fit(z, y, rhs, tr, pl):
    z = z.dropna(subset=[y]).copy()
    if pl: z = z[z.t < 0]; z["post"] = (z.t >= -12).astype(int)
    return smf.wls(f"{y} ~ C(site) + C(month) + {rhs}" + (" + C(site):t" if tr else ""), z, weights=np.minimum(z.n, 2000)).fit(cov_type="cluster", cov_kwds={"groups": z.site.astype("category").cat.codes})
out = []
specs = {"edit": "r_edit:post", "code+edit": "rank:post + r_edit:post", "all5": "rank:post + r_edit:post + r_size:post + r_len:post + r_math:post"}
for y in ["ext", "anyx", "list", "head", "lw"]:
    for lab, rhs in specs.items():
        for tr in [False, True]:
            for pl in [False, True]:
                m = fit(P, y, rhs, tr, pl)
                for k in [k for k in m.params.index if k.endswith(":post")]: out.append(dict(outcome=y, spec=lab, trend=tr, placebo=pl, term=k[:-5], b=m.params[k], se=m.bse[k], p=m.pvalues[k]))
    print(y, flush=True)
O = pd.DataFrame(out); O.to_csv(R + r"/edit_dose.csv", index=False); pd.set_option("display.width", 220); pd.set_option("display.max_rows", 400)
print(O[O.term.isin(["r_edit", "rank"])].round(4).to_string())
# event study joint code + edit, no trend
P["yr"] = P.month.str[:4] + np.where(P.month.str[5:7].astype(int) <= 6, "H1", "H2"); terms = []
for pp_ in sorted(P.yr.unique()):
    if pp_ == "2022H2": continue
    for d in ["rank", "r_edit"]: c = f"{d}_{pp_}"; P[c] = (P.yr == pp_) * P[d]; terms.append(c)
rows = []
for y in ["ext", "anyx"]:
    m = smf.wls(f"{y} ~ C(site) + C(month) + " + " + ".join(terms), P, weights=np.minimum(P.n, 2000)).fit(cov_type="cluster", cov_kwds={"groups": P.site.astype("category").cat.codes})
    for t in terms: d, yy = t.rsplit("_", 1); rows.append(dict(outcome=y, dose=d, yr=yy, b=m.params[t], se=m.bse[t]))
    for d in ["rank", "r_edit"]: rows.append(dict(outcome=y, dose=d, yr="2022H2", b=0., se=0.))
EV = pd.DataFrame(rows); EV.to_csv(R + r"/event_code_edit.csv", index=False); print(EV.pivot_table(index="yr", columns=["outcome", "dose"], values="b").round(4).to_string())
print(EV.pivot_table(index="yr", columns=["outcome", "dose"], values="se").round(4).to_string())
# leave crypto out
crypto = ["ethereum", "bitcoin", "eosio", "tezos", "solana", "cardano", "monero", "iota", "stellar", "substrate", "crypto", "tron", "vi"]
cr = [c for c in crypto if c in set(E.site)]; print("crypto present", cr)
for tr in [False, True]:
    m = fit(P[~P.site.isin(cr)], "ext", "rank:post + r_edit:post", tr, False); print("no crypto, trend", tr, {k: (round(m.params[k], 4), round(m.pvalues[k], 4)) for k in ["rank:post", "r_edit:post"]})
# RI for edit single no trend
z = P.reset_index(drop=True); w = np.sqrt(np.minimum(z.n, 2000).values)
X = np.hstack([pd.get_dummies(z.site).values, pd.get_dummies(z.month).values[:, 1:]]).astype(float) * w[:, None]; Q, _ = np.linalg.qr(X); res = lambda v: v - Q @ (Q.T @ v)
us = np.unique(z.site); pos = np.searchsorted(us, z.site.values)
for y in ["ext", "anyx"]:
    yt = res(z[y].values * w)
    for d in ["r_edit", "rank"]:
        rk = z.drop_duplicates("site").set_index("site")[d].reindex(us).values; beta = lambda r: (lambda x: (x @ yt) / (x @ x))(res(r[pos] * z.post.values * w))
        b0 = beta(rk); g = np.random.default_rng(4); bs = np.array([beta(g.permutation(rk)) for _ in range(2000)]); print("RI notrend", y, d, round(b0, 5), "p", (np.abs(bs) >= abs(b0)).mean())
