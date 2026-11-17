"""Aggregate level test with frequency-matched random lists. For each list: site-quarter score = mean over words of 100*df/n;
per tercile and overall, equal-weight mean over sites with n>=20; ratio = mean(2023Q2..2024Q1) / linear extrapolation of 2019Q1..2022Q3.
Also per-quarter ratio path of marker list vs null band. Also leave-ensure-out marker list."""
import os, numpy as np, pandas as pd
H = str(__import__("pathlib").Path(__file__).resolve().parents[1]); V = H + r"/vocab"; R = H + r"/results_vocab"
src = open(H + r"/code/07_vocab.py", encoding="utf8").read(); i = src.index("EXT = set("); j = src.index(".split())", i) + 9; EXT = sorted(eval(src[i + 6:j]))
E = pd.read_csv(H + r"/results_all/exposure.csv"); E["ter"] = pd.qcut(E.code_q.rank(method="first"), 3, labels=["low", "mid", "high"]).astype(str)
QS = [f"{y}Q{k}" for y in range(2019, 2025) for k in range(1, 5) if f"{y}Q{k}" <= "2024Q1"]; FIT = [q for q in QS if "2021Q1" <= q <= "2022Q3"]
FITL = [q for q in QS if q <= "2022Q3"]; OBS = ["2023Q2", "2023Q3", "2023Q4", "2024Q1"]
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
keep = set(pool.index) | set(EXT)
W = V_[V_.word.isin(keep)].pivot_table(index=["site", "q"], columns="word", values="df", aggfunc="sum", fill_value=0)
base = N.set_index(["site", "q"]).n; base = base[base >= 20]; W = W.reindex(base.index, fill_value=0); Sh = 100 * W.div(base, axis=0)
ter = base.index.get_level_values("site").map(E.set_index("site").ter); qq = base.index.get_level_values("q")
tq = np.array([QS.index(q) for q in qq])
def ratio(score):
    out = {}
    for k in ["all", "low", "mid", "high"]:
        m = np.ones(len(score), bool) if k == "all" else (ter == k)
        s = pd.Series(score[m]).groupby(qq[m]).mean().reindex(QS); t = np.arange(len(QS)); f = np.isin(QS, FITL)
        b = np.polyfit(t[f], s.values[f], 1); r = s.values / np.polyval(b, t); out[k] = r[np.isin(QS, OBS)].mean(); out[k + "_path"] = r
    return out
cols = list(Sh.columns); ci = {w: i for i, w in enumerate(cols)}; A = Sh.values
def score(lst): idx = [ci[w] for w in lst if w in ci]; return A[:, idx].mean(axis=1)
mlist = [w for w in EXT if w in ci]; m0 = ratio(score(mlist)); m1 = ratio(score([w for w in mlist if w != "ensure"]))
rng = np.random.default_rng(99); pw = pool.index.values; null = []; paths = []
for r in range(1000):
    lst = [rng.choice(pw[pb == b_]) for b_ in mb]; o = ratio(score(lst)); null.append({k: o[k] for k in ["all", "low", "mid", "high"]}); paths.append(o["all_path"])
Nl = pd.DataFrame(null); Nl.to_csv(R + r"/aggnull.csv", index=False)
paths = np.array(paths); band = pd.DataFrame({"q": QS, "marker": m0["all_path"], "noensure": m1["all_path"], "lo": np.quantile(paths, .025, axis=0), "hi": np.quantile(paths, .975, axis=0), "med": np.median(paths, axis=0),
                                             "m_high": m0["high_path"], "m_mid": m0["mid_path"], "m_low": m0["low_path"]})
band.to_csv(R + r"/aggnull_path.csv", index=False)
for k in ["all", "low", "mid", "high"]:
    print(k, "marker", round(m0[k], 3), "no-ensure", round(m1[k], 3), "null median", round(Nl[k].median(), 3), "q95", round(Nl[k].quantile(.95), 3), "q99", round(Nl[k].quantile(.99), 3), "max", round(Nl[k].max(), 3), "share>=", (Nl[k] >= m0[k]).mean())
print(band.round(3).to_string())
