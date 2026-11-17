"""Site-quarter word panel. (1) per-word dose-response for the fixed marker list; (2) permutation null: 500 random lists of
non-marker words matched to the marker list on pooled 2021Q1-2022Q3 document share (frequency deciles), same regression.
Score = mean over list words of 100 * df/n. Spec: score ~ C(site)+C(q)+rank:post (+C(site):t), WLS n capped, cluster site."""
import glob, os, numpy as np, pandas as pd, statsmodels.formula.api as smf
H = str(__import__("pathlib").Path(__file__).resolve().parents[1]); V = H + r"/vocab"; R = H + r"/results_vocab"
src = open(H + r"/code/07_vocab.py", encoding="utf8").read(); i = src.index("EXT = set("); j = src.index(".split())", i) + 9; EXT = sorted(eval(src[i + 6:j]))
E = pd.read_csv(H + r"/results_all/exposure.csv"); E["rank"] = E.code_q.rank(pct=True) - .5
QS = [f"{y}Q{k}" for y in range(2019, 2025) for k in range(1, 5) if f"{y}Q{k}" <= "2024Q1"]
FIT = [q for q in QS if "2021Q1" <= q <= "2022Q3"]
vs, ns = [], []
for s in E.site:
    f = V + rf"/vocab_{s}.parquet"
    if not os.path.exists(f): continue
    v = pd.read_parquet(f); v = v[v.q.isin(QS)]; v["site"] = s; vs.append(v)
    n = pd.read_parquet(V + rf"/nq_{s}.parquet"); n = n[n.q.isin(QS)]; n["site"] = s; ns.append(n)
N = pd.concat(ns); V_ = pd.concat(vs)
# candidate pool: words present in >= 50 sites in the fit window, alphabetic, len >= 4
pre = V_[V_.q.isin(FIT)]; nsites = pre.groupby("word").site.nunique()
tot = pre.groupby("word").df.sum() / N[N.q.isin(FIT)].n.sum()
pool = tot[(nsites >= 50) & tot.index.str.fullmatch(r"[a-z]{4,}") & ~tot.index.isin(EXT)]
mk = tot.reindex([w for w in EXT if w in tot.index]).dropna()
bins = np.quantile(pool.values, np.linspace(0, 1, 21)); pb = np.clip(np.digitize(pool.values, bins[1:-1]), 0, 19); mb = np.clip(np.digitize(mk.values, bins[1:-1]), 0, 19)
print("pool", len(pool), "markers present", len(mk))
keep = set(pool.index) | set(EXT)
W = V_[V_.word.isin(keep)].pivot_table(index=["site", "q"], columns="word", values="df", aggfunc="sum", fill_value=0)
base = N.set_index(["site", "q"]).n; idx = pd.MultiIndex.from_product([E.site[E.site.isin(N.site)], QS], names=["site", "q"])
base = base.reindex(idx); W = W.reindex(idx, fill_value=0); ok = base.notna() & (base >= 20); W = W[ok.values]; base = base[ok]
Sh = 100 * W.div(base, axis=0)
P0 = pd.DataFrame({"n": base}).reset_index(); P0["t"] = P0.q.map(lambda q: (int(q[:4]) - 2022) * 12 + (int(q[5]) - 1) * 3 + 1 - 12)
P0 = P0.merge(E[["site", "rank"]], on="site", how="left"); P0["post"] = (P0.t >= 0).astype(int)
P0["site_c"] = P0.site.astype("category").cat.codes
def fit(y, trend, placebo=False):
    z = P0.assign(y=y.values)
    if placebo: z = z[z.t < 0].copy(); z["post"] = (z.t >= -12).astype(int)
    m = smf.wls("y ~ C(site) + C(q) + rank:post" + (" + C(site):t" if trend else ""), z, weights=np.minimum(z.n, 2000)).fit(
        cov_type="cluster", cov_kwds={"groups": z.site_c})
    return m.params["rank:post"], m.bse["rank:post"], m.pvalues["rank:post"]
# marker list score, and per word
out = []
mscore = Sh[[w for w in EXT if w in Sh.columns]].mean(axis=1)
for tr in [False, True]:
    for pl in [False, True]:
        b, se, p = fit(mscore, tr, pl); out.append(dict(word="__LIST__", trend=tr, placebo=pl, b=b, se=se, p=p))
for w in [w for w in EXT if w in Sh.columns]:
    for pl in [False, True]:
        b, se, p = fit(Sh[w], True, pl); out.append(dict(word=w, trend=True, placebo=pl, b=b, se=se, p=p, pre=float(tot.get(w, np.nan)) * 100))
O = pd.DataFrame(out); O.to_csv(R + r"/word_dose.csv", index=False)
# relative effect: list b / pre mean
pre_mean = mscore[P0.set_index(["site", "q"]).index.get_level_values("q").isin(FIT)].mean()
# permutation null with frequency-matched lists
rng = np.random.default_rng(2026); null = []
pw = pool.index.values
for r in range(500):
    lst = [rng.choice(pw[pb == b_]) for b_ in mb]
    sc = Sh[[w for w in lst if w in Sh.columns]].mean(axis=1)
    b, se, p = fit(sc, True); sc_rel = b / sc[P0.q.isin(FIT).values].mean()
    null.append(dict(rep=r, b=b, rel=sc_rel))
    if r % 50 == 0: print("perm", r, flush=True)
Nl = pd.DataFrame(null); Nl.to_csv(R + r"/perm_null.csv", index=False)
bm = O[(O.word == "__LIST__") & O.trend & ~O.placebo].b.iloc[0]; rel = bm / pre_mean
print(O[O.word == "__LIST__"].round(5).to_string()); print("pre mean", pre_mean, "rel", rel)
print("perm: share of null rel >= marker rel", (Nl.rel >= rel).mean(), "null rel q95", Nl.rel.quantile(.95), "q99", Nl.rel.quantile(.99))
pd.set_option("display.max_rows", 200); print(O[(O.word != "__LIST__") & ~O.placebo].sort_values("b", ascending=False).round(5).to_string())
