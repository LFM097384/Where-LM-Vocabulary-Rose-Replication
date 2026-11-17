"""Robustness of the network-wide rise. Ratio of 2023Q2-2024Q1 score to linear (or quadratic) trend.
Variants: fit window 2019Q1-2022Q3 linear (baseline), 2021Q1-2022Q3 linear, 2019Q1-2022Q3 quadratic; equal-site vs answer-weighted;
controls: frequency-matched random lists (200), register lists (formal connectives; formal adjectives/adverbs) not on marker list.
Also: site screen saved (edit, code, joint), drop small sites, word-level edit gradient, and screen without 'ensure'."""
import os, numpy as np, pandas as pd, statsmodels.formula.api as smf
H = str(__import__("pathlib").Path(__file__).resolve().parents[1]); V = H + r"/vocab"; R = H + r"/results_vocab"
src = open(H + r"/code/07_vocab.py", encoding="utf8").read(); i = src.index("EXT = set("); j = src.index(".split())", i) + 9; EXT = sorted(eval(src[i + 6:j]))
E = pd.read_csv(R + r"/exposure_plus.csv")
CONN = "however therefore thus hence consequently nevertheless accordingly whereas although indeed likewise subsequently respectively nonetheless".split()
FORM = "significant significantly specifically particularly approximately typically generally primarily necessary appropriate relevant sufficient considerable adequate various numerous efficient effective potential important".split()
QS = [f"{y}Q{k}" for y in range(2019, 2025) for k in range(1, 5) if f"{y}Q{k}" <= "2024Q1"]; FIT = [q for q in QS if "2021Q1" <= q <= "2022Q3"]; OBS = ["2023Q2", "2023Q3", "2023Q4", "2024Q1"]
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
keep = set(pool.index) | set(EXT) | set(CONN) | set(FORM)
W = V_[V_.word.isin(keep)].pivot_table(index=["site", "q"], columns="word", values="df", aggfunc="sum", fill_value=0)
base = N.set_index(["site", "q"]).n; base = base[base >= 20]; W = W.reindex(base.index, fill_value=0); Sh = 100 * W.div(base, axis=0)
ci = {w: k for k, w in enumerate(Sh.columns)}; A = Sh.values; sc = lambda l: A[:, [ci[w] for w in l if w in ci]].mean(axis=1)
print("conn present", [w for w in CONN if w in ci]); print("form present", [w for w in FORM if w in ci])
qq = base.index.get_level_values("q"); nn = base.values; tq = np.arange(len(QS))
def ratio(score, weighted=False, fit="lin19"):
    if weighted: s = pd.Series(score * nn).groupby(qq).sum() / pd.Series(nn).groupby(qq).sum()
    else: s = pd.Series(score).groupby(qq).mean()
    s = s.reindex(QS).values
    if fit == "lin19": f = np.array([q <= "2022Q3" for q in QS]); deg = 1
    elif fit == "lin21": f = np.isin(QS, FIT); deg = 1
    else: f = np.array([q <= "2022Q3" for q in QS]); deg = 2
    b = np.polyfit(tq[f], s[f], deg); r = s / np.polyval(b, tq); return r[np.isin(QS, OBS)].mean()
rng = np.random.default_rng(7); pw = pool.index.values; ctl = [sc([rng.choice(pw[pb == b_]) for b_ in mb]) for _ in range(200)]
mlist = [w for w in EXT if w in ci]; out = []
for wt in [False, True]:
    for fit in ["lin19", "lin21", "quad19"]:
        cr = np.array([ratio(c, wt, fit) for c in ctl])
        out.append(dict(weighted=wt, fit=fit, marker=ratio(sc(mlist), wt, fit), marker_noensure=ratio(sc([w for w in mlist if w != "ensure"]), wt, fit),
                        ctl_median=np.median(cr), ctl_max=cr.max(), ctl_q975=np.quantile(cr, .975), connectives=ratio(sc(CONN), wt, fit), formal=ratio(sc(FORM), wt, fit)))
O = pd.DataFrame(out); O.to_csv(R + r"/rise_robust.csv", index=False); pd.set_option("display.width", 220); print(O.round(3).to_string())
# per-word ratios for register words
print({w: round(ratio(A[:, ci[w]]), 3) for w in CONN + FORM if w in ci})
