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
Wp = Wr[np.isfinite(Wr.r_pl_lo) & (Wr.pre_lo_per1000 >= 2) & (Wr.pre_hi_per1000 >= 2)].sort_values("r_lo"); print(Wp.round(2).to_string())
fig, ax = plt.subplots(figsize=(3.3, 4.2)); y = np.arange(len(Wp))
ax.scatter(Wp.r_lo, y, s=9, color="#b2182b", label="least edited third, 2023 vs 2021--22", zorder=3)
ax.scatter(Wp.r_hi, y, s=9, facecolors="none", edgecolors="#2166ac", label="most edited third, 2023 vs 2021--22", zorder=3)
ax.scatter(Wp.r_pl_lo, y, s=7, marker="x", color="#888888", label="least edited third, placebo 2021--22 vs 2019--20", zorder=2)
ax.axvline(1, color="grey", lw=.5); ax.set_yticks(y); ax.set_yticklabels(Wp.index, fontsize=5.6); ax.set_xlabel("ratio of document share")
ax.legend(fontsize=5, frameon=False, loc="lower right"); plt.tight_layout(); plt.savefig(O + r"/fig_words.pdf"); plt.savefig(O + r"/fig_words.png", dpi=160); plt.close()
print("tercile sizes", E.ter.value_counts().to_dict())

