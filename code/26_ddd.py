"""Triple difference. Control vocabulary = union of 200 frequency-matched random lists (score = mean of their scores).
Site-quarter outcome y = marker_score/mk_pre - control_score/ctl_pre, pre = pooled 2021Q1-2022Q3 mean across all sites (so y is in
'share of base rate' units). Dose regressions: single code rank; joint with size, length, math, edited ranks; trends; placebo (break 2021Q4).
RI permutation of code rank."""
import os, numpy as np, pandas as pd, statsmodels.formula.api as smf
H = str(__import__("pathlib").Path(__file__).resolve().parents[1]); V = H + r"/vocab"; R = H + r"/results_vocab"
src = open(H + r"/code/07_vocab.py", encoding="utf8").read(); i = src.index("EXT = set("); j = src.index(".split())", i) + 9; EXT = sorted(eval(src[i + 6:j]))
E = pd.read_csv(H + r"/results_all/exposure.csv"); E["rank"] = E.code_q.rank(pct=True) - .5
pn = pd.read_csv(H + r"/results_all/panel.csv"); pp = pn[pn.month.between("2021-01", "2022-11")]
E = E.merge(pp.groupby("site").apply(lambda g: np.average(g.edited.fillna(0), weights=g.n.fillna(0) + 1e-9)).rename("edited").reset_index(), on="site")
for c, s_ in [("r_size", "n_pre"), ("r_len", "lw_pre"), ("r_math", "math_a"), ("r_edit", "edited")]: E[c] = E[s_].rank(pct=True) - .5
QS = [f"{y}Q{k}" for y in range(2019, 2025) for k in range(1, 5) if f"{y}Q{k}" <= "2024Q1"]; FIT = [q for q in QS if "2021Q1" <= q <= "2022Q3"]
vs, ns = [], []
for s in E.site:
    f = V + rf"/vocab_{s}.parquet"
    if not os.path.exists(f): continue
    v = pd.read_parquet(f); v = v[v.q.isin(QS)]; v["site"] = s; vs.append(v)
    n = pd.read_parquet(V + rf"/nq_{s}.parquet"); n = n[n.q.isin(QS)]; n["site"] = s; ns.append(n)
N = pd.concat(ns); V_ = pd.concat(vs)
pre = V_[V_.q.isin(FIT)]; nsites = pre.groupby("word").site.nunique(); tot = pre.groupby("word").df.sum() / N[N.q.isin(FIT)].n.sum()
pool = tot[(nsites >= 50) & tot.index.str.fullmatch(r"[a-z]{4,}") & ~tot.index.isin(EXT)]; mk = tot.reindex([w for w in EXT if w in tot.index]).dropna()
bins = np.quantile(pool.values, np.linspace(0, 1, 21)); pb = np.clip(np.digitize(pool.values, bins[1:-1]), 0, 19); mb = np.clip(np.digitize(mk.values, bins[1:-1]), 0, 19)
W = V_[V_.word.isin(set(pool.index) | set(EXT))].pivot_table(index=["site", "q"], columns="word", values="df", aggfunc="sum", fill_value=0)
base = N.set_index(["site", "q"]).n; base = base[base >= 20]; W = W.reindex(base.index, fill_value=0); Sh = 100 * W.div(base, axis=0)
ci = {w: k for k, w in enumerate(Sh.columns)}; A = Sh.values; sc = lambda l: A[:, [ci[w] for w in l if w in ci]].mean(axis=1)
rng = np.random.default_rng(5); pw = pool.index.values
ctl = np.mean([sc([rng.choice(pw[pb == b_]) for b_ in mb]) for _ in range(200)], axis=0); mks = sc([w for w in EXT if w in ci])
mks_ne = sc([w for w in EXT if w in ci and w != "ensure"])
P = base.rename("n").reset_index(); fitm = P.q.isin(FIT).values
P["mk"] = mks / mks[fitm].mean(); P["ctl"] = ctl / ctl[fitm].mean(); P["y"] = P.mk - P.ctl; P["y_ne"] = mks_ne / mks_ne[fitm].mean() - P.ctl
P["t"] = P.q.map(lambda q: (int(q[:4]) - 2022) * 4 + int(q[5]) - 4); P = P.merge(E[["site", "rank", "r_size", "r_len", "r_math", "r_edit"]], on="site"); P["post"] = (P.t >= 0).astype(int)
P.to_csv(R + r"/ddd_panel.csv", index=False)
J = "rank:post + r_size:post + r_len:post + r_math:post + r_edit:post"
out = []
for y in ["y", "y_ne", "mk", "ctl"]:
    for rhs, lab in [("rank:post", "single"), (J, "joint")]:
        for tr in [False, True]:
            for pl in [False, True]:
                z = P.copy()
                if pl: z = z[z.t < 0]; z["post"] = (z.t >= -4).astype(int)
                m = smf.wls(f"{y} ~ C(site) + C(q) + {rhs}" + (" + C(site):t" if tr else ""), z, weights=np.minimum(z.n, 6000)).fit(cov_type="cluster", cov_kwds={"groups": z.site.astype("category").cat.codes})
                out.append(dict(outcome=y, model=lab, trend=tr, placebo=pl, b=m.params["rank:post"], se=m.bse["rank:post"], p=m.pvalues["rank:post"],
                                b_size=m.params.get("r_size:post", np.nan), b_len=m.params.get("r_len:post", np.nan), b_edit=m.params.get("r_edit:post", np.nan)))
O = pd.DataFrame(out); O.to_csv(R + r"/ddd.csv", index=False); pd.set_option("display.width", 220); print(O.round(4).to_string())
# event study, ddd, joint controls interacted by year
P["yr"] = P.q.str[:4]; P.loc[P.q.isin(["2022Q4"]), "yr"] = "2022Q4"; terms = []
for yy in sorted(P.yr.unique()):
    if yy == "2022": continue
    for d in ["rank", "r_size", "r_len", "r_edit"]:
        c = f"{d}_{yy}"; P[c] = (P.yr == yy) * P[d]; terms.append(c)
m = smf.wls("y ~ C(site) + C(q) + " + " + ".join(terms), P, weights=np.minimum(P.n, 6000)).fit(cov_type="cluster", cov_kwds={"groups": P.site.astype("category").cat.codes})
EV = pd.DataFrame([dict(term=t, b=m.params[t], se=m.bse[t]) for t in terms if t.startswith("rank_")]); EV.to_csv(R + r"/ddd_event.csv", index=False); print(EV.round(4).to_string())
# RI single no-trend and joint no-trend
for rhs_extra in [[], ["r_size", "r_len", "r_math", "r_edit"]]:
    z = P.reset_index(drop=True); w = np.sqrt(np.minimum(z.n, 6000).values)
    X = np.hstack([pd.get_dummies(z.site).values, pd.get_dummies(z.q).values[:, 1:]] + ([z[rhs_extra].values * z.post.values[:, None]] if rhs_extra else [])).astype(float) * w[:, None]
    Q, _ = np.linalg.qr(X); res = lambda v: v - Q @ (Q.T @ v); yt = res(z.y.values * w)
    us = np.unique(z.site); pos = np.searchsorted(us, z.site.values); rk = z.drop_duplicates("site").set_index("site")["rank"].reindex(us).values
    beta = lambda r: (lambda x: (x @ yt) / (x @ x))(res(r[pos] * z.post.values * w))
    b0 = beta(rk); g = np.random.default_rng(3); bs = np.array([beta(g.permutation(rk)) for _ in range(2000)])
    print("RI ddd", "joint" if rhs_extra else "single", round(b0, 4), "p", (np.abs(bs) >= abs(b0)).mean())
