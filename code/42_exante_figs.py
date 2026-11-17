"""Figures rebuilt around the ex-ante edit share (r_ea30 = share of 2021-01..2022-10 answers edited within 30 days of posting,
every such edit precedes the release). (1) null band + marker path by edit tercile; (2) half-year event studies: edit alone,
code alone, code and edit together; (3) site scatter; (4) word ratios least vs most edited tercile, with placebo."""
import os, numpy as np, pandas as pd, statsmodels.formula.api as smf, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
H = str(__import__("pathlib").Path(__file__).resolve().parents[1]); V = H + r"/vocab"; R = H + r"/results_vocab"; O = H + r"/paper"
plt.rcParams.update({"font.size": 7, "font.family": "serif", "axes.spines.top": False, "axes.spines.right": False})
S = pd.read_csv(R + r"/exante_edit.csv").set_index("site")
src = open(H + r"/code/07_vocab.py", encoding="utf8").read(); i = src.index("EXT = set("); j = src.index(".split())", i) + 9; EXT = sorted(eval(src[i + 6:j]))
E = pd.read_csv(H + r"/results_all/exposure.csv"); E = E[E.site.isin(S.index)].copy()
E["ter"] = pd.qcut(E.site.map(S.ea30).rank(method="first"), 3, labels=["low", "mid", "high"]).astype(str)
QS = [f"{y}Q{k}" for y in range(2019, 2025) for k in range(1, 5) if f"{y}Q{k}" <= "2024Q1"]; FIT = [q for q in QS if "2021Q1" <= q <= "2022Q3"]
FITL = [q for q in QS if q <= "2022Q3"]; OBS = ["2023Q2", "2023Q3", "2023Q4", "2024Q1"]
# ---------- (1) null band by edit tercile
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
def ratio(score):
    out = {}
    for k in ["all", "low", "mid", "high"]:
        m = np.ones(len(score), bool) if k == "all" else (ter == k)
        s = pd.Series(score[m]).groupby(qq[m]).mean().reindex(QS); t = np.arange(len(QS)); f = np.isin(QS, FITL)
        b = np.polyfit(t[f], s.values[f], 1); r = s.values / np.polyval(b, t); out[k] = r[np.isin(QS, OBS)].mean(); out[k + "_path"] = r
    return out
cols = list(Sh.columns); ci = {w: i for i, w in enumerate(cols)}; A = Sh.values
score = lambda lst: A[:, [ci[w] for w in lst if w in ci]].mean(axis=1)
m0 = ratio(score([w for w in EXT if w in ci]))
rng = np.random.default_rng(99); pw = pool.index.values; null, paths = [], []
for r in range(1000):
    o = ratio(score([rng.choice(pw[pb == b_]) for b_ in mb])); null.append({k: o[k] for k in ["all", "low", "mid", "high"]}); paths.append(o["all_path"])
Nl = pd.DataFrame(null); Nl.to_csv(R + r"/aggnull_edit.csv", index=False); paths = np.array(paths)
B = pd.DataFrame({"q": QS, "marker": m0["all_path"], "lo": np.quantile(paths, .025, axis=0), "hi": np.quantile(paths, .975, axis=0),
                  "m_high": m0["high_path"], "m_mid": m0["mid_path"], "m_low": m0["low_path"]}); B.to_csv(R + r"/aggnull_edit_path.csv", index=False)
for k in ["all", "low", "mid", "high"]:
    print("NULL", k, "marker", round(m0[k], 3), "null med", round(Nl[k].median(), 3), "q2.5", round(Nl[k].quantile(.025), 3), "q97.5", round(Nl[k].quantile(.975), 3), "max", round(Nl[k].max(), 3), "share>=", (Nl[k] >= m0[k]).mean())
fig, (a, b) = plt.subplots(1, 2, figsize=(7, 2.3), gridspec_kw={"width_ratios": [2.2, 1]})
x = np.arange(len(B)); a.fill_between(x, B.lo, B.hi, color="0.85", lw=0, label="matched random lists, 95% band")
a.plot(x, B.marker, color="k", lw=1.4, label="marker list, all sites"); a.plot(x, B.m_low, color="#b2182b", lw=1.1, label="least edited third")
a.plot(x, B.m_mid, color="0.45", lw=.9, ls="--", label="middle third"); a.plot(x, B.m_high, color="#2166ac", lw=.9, ls=":", label="most edited third")
a.axhline(1, color="k", lw=.4); a.axvline(15.67, color="k", lw=.5, ls="--"); a.axvspan(-.5, 14.5, color="0.97", lw=0, zorder=0)
a.set_xticks(x[::4]); a.set_xticklabels(B.q[::4]); a.set_ylabel("document share / pre-2022Q4 trend"); a.legend(frameon=False, fontsize=5.5, loc="upper left"); a.set_xlim(-.5, len(B) - .5)
lab = {"low": "least\nedited", "mid": "middle", "high": "most\nedited", "all": "all"}
for i_, k in enumerate(["low", "mid", "high", "all"]):
    v = Nl[k].values; b.scatter(np.full(len(v), i_) + np.random.default_rng(i_).uniform(-.18, .18, len(v)), v, s=1, color="0.6", lw=0)
    b.scatter([i_], [m0[k]], marker="D", s=14, color="#b2182b" if k != "all" else "k", zorder=3)
b.set_xticks(range(4)); b.set_xticklabels([lab[k] for k in ["low", "mid", "high", "all"]], fontsize=6); b.axhline(1, color="k", lw=.4); b.set_ylabel("2023Q2--2024Q1 ratio to trend")
plt.tight_layout(); plt.savefig(O + r"/fig_null.pdf"); plt.savefig(O + r"/fig_null.png", dpi=160); plt.close()
# ---------- (2) event studies
P = pd.read_parquet(R + r"/panel_edit.parquet").merge(S[["r_ea30"]].reset_index(), on="site")
P["yr"] = P.month.str[:4] + np.where(P.month.str[5:7].astype(int) <= 6, "H1", "H2"); yrs = sorted(P.yr.unique())
def es(doses):
    terms = []
    for p_ in yrs:
        if p_ == "2022H2": continue
        for d in doses: c = f"{d}_{p_}"; P[c] = (P.yr == p_) * P[d]; terms.append(c)
    m = smf.wls("ext ~ C(site) + C(month) + " + " + ".join(terms), P, weights=np.minimum(P.n, 2000)).fit(cov_type="cluster", cov_kwds={"groups": P.site.astype("category").cat.codes})
    rows = [dict(dose=t.rsplit("_", 1)[0], yr=t.rsplit("_", 1)[1], b=m.params[t], se=m.bse[t]) for t in terms] + [dict(dose=d, yr="2022H2", b=0., se=0.) for d in doses]
    return pd.DataFrame(rows)
e1 = es(["r_ea30"]); e2 = es(["rank"]); e3 = es(["rank", "r_ea30"])
pd.concat([e1.assign(spec="edit"), e2.assign(spec="code"), e3.assign(spec="joint")]).to_csv(R + r"/event_exante.csv", index=False)
print(e1.sort_values("yr").round(4).to_string()); print(e3.pivot_table(index="yr", columns="dose", values="b").round(4).to_string())
fig, ax = plt.subplots(1, 3, figsize=(7, 2.0), sharey=True)
def panel(a, e, col, lab_, off=0, mk_="o"):
    e = e.sort_values("yr").reset_index(drop=True); xx = np.arange(len(e)) + off
    a.errorbar(xx, e.b, yerr=1.96 * e.se, fmt=mk_, ms=2.6, color=col, lw=.7, capsize=1.2, label=lab_); return e
e = panel(ax[0], e1, "#b2182b", "edit-share rank", mk_="s"); ax[0].set_title("edit share alone", fontsize=6.8)
panel(ax[1], e2, "k", "code rank"); ax[1].set_title("code exposure alone", fontsize=6.8)
panel(ax[2], e3[e3.dose == "rank"], "k", "code rank", -.15); panel(ax[2], e3[e3.dose == "r_ea30"], "#b2182b", "edit-share rank", .15, "s")
ax[2].set_title("both together", fontsize=6.8); ax[2].legend(frameon=False, fontsize=5.5, loc="lower left")
for a in ax:
    a.axhline(0, color="grey", lw=.5); a.axvline(list(e.yr).index("2022H2") + .5, color="#b2182b", ls="--", lw=.7)
    a.set_xticks(np.arange(len(e))[::2]); a.set_xticklabels(e.yr[::2], rotation=45, fontsize=5.5)
ax[0].set_ylabel("marker rate, slope on centred rank", fontsize=6.5)
plt.tight_layout(); plt.savefig(O + r"/fig_event.pdf"); plt.savefig(O + r"/fig_event.png", dpi=160); plt.close()
# ---------- (3) site scatter on ex-ante edit share
d = pd.read_csv(R + r"/site_logchange2.csv").set_index("site")[["y", "w", "rank"]].join(S[["ea30"]], how="inner")
fig, ax = plt.subplots(figsize=(3.3, 2.4)); c = np.where(d["rank"] > 1 / 6, "#b2182b", np.where(d["rank"] < -1 / 6, "#2166ac", "0.55"))
ax.scatter(d.ea30, d.y, s=d.w / d.w.max() * 40 + 3, c=c, alpha=.7, lw=0)
for l_, col in [("high code exposure", "#b2182b"), ("middle", "0.55"), ("low code exposure", "#2166ac")]: ax.scatter([], [], c=col, s=12, label=l_)
bb = np.polyfit(d.ea30, d.y, 1, w=d.w); xx = np.linspace(d.ea30.min(), d.ea30.max(), 10); ax.plot(xx, np.polyval(bb, xx), color="k", lw=.8)
ax.axhline(0, color="grey", lw=.4); ax.set_xlabel("share of 2021--22 answers edited within 30 days"); ax.set_ylabel("marker change net of control (log)")
for s in ["robotics", "ethereum", "magento", "skeptics", "history", "rpg", "codereview"]:
    if s in d.index: ax.annotate(s, (d.ea30[s], d.y[s]), fontsize=5, xytext=(2, 1), textcoords="offset points")
ax.legend(frameon=False, fontsize=5.5, loc="upper right"); plt.tight_layout(); plt.savefig(O + r"/fig_edit.pdf"); plt.savefig(O + r"/fig_edit.png", dpi=160); plt.close()
# ---------- (4) word ratios by edit tercile
st = E.set_index("site").ter; rows = []
for s in E.site:
    f = V + rf"/vocab_{s}.parquet"
    if not os.path.exists(f): continue
    v = pd.read_parquet(f); n = pd.read_parquet(V + rf"/nq_{s}.parquet").set_index("q").n
    v = v[v.word.isin(EXT)].copy(); v["sh"] = v.df / v.q.map(n)
    g = lambda a_, b_: v[v.q.between(a_, b_)].groupby("word").sh.mean()
    dd = pd.DataFrame({"pre0": g("2019Q1", "2020Q3"), "pre": g("2021Q1", "2022Q3"), "post": g("2023Q2", "2024Q1")}).fillna(0); dd["site"] = s; dd["ter"] = st[s]; rows.append(dd.reset_index())
D = pd.concat(rows); g = D.groupby(["word", "ter"])[["pre0", "pre", "post"]].mean().unstack()
Wr = pd.DataFrame({"r_lo": g[("post", "low")] / g[("pre", "low")], "r_hi": g[("post", "high")] / g[("pre", "high")], "r_pl_lo": g[("pre", "low")] / g[("pre0", "low")],
                   "pre_lo_per1000": 1000 * g[("pre", "low")], "pre_hi_per1000": 1000 * g[("pre", "high")]})
Wr.to_csv(R + r"/word_ratios_edit.csv")
Wp = Wr[np.isfinite(Wr.r_pl_lo) & (Wr.pre_lo_per1000 >= .5) & (Wr.pre_hi_per1000 > 0)].sort_values("r_lo"); print(Wp.round(2).to_string())
fig, ax = plt.subplots(figsize=(3.3, 4.2)); y = np.arange(len(Wp))
ax.scatter(Wp.r_lo, y, s=9, color="#b2182b", label="least edited third, 2023 vs 2021--22", zorder=3)
ax.scatter(Wp.r_hi, y, s=9, facecolors="none", edgecolors="#2166ac", label="most edited third, 2023 vs 2021--22", zorder=3)
ax.scatter(Wp.r_pl_lo, y, s=7, marker="x", color="#888888", label="least edited third, placebo 2021--22 vs 2019--20", zorder=2)
ax.axvline(1, color="grey", lw=.5); ax.set_yticks(y); ax.set_yticklabels(Wp.index, fontsize=5.6); ax.set_xlabel("ratio of document share")
ax.legend(fontsize=5, frameon=False, loc="lower right"); plt.tight_layout(); plt.savefig(O + r"/fig_words.pdf"); plt.savefig(O + r"/fig_words.png", dpi=160); plt.close()
print("tercile sizes", E.ter.value_counts().to_dict())
