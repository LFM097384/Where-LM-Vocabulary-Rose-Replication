"""Marker rate by exposure tercile, equal-weight over sites (n>=20), half-years, relative to own 2019-01..2022-11 linear trend.
Plus a figure: aggregate (all sites) and tercile series, observed / trend-extrapolated."""
import glob, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
H = str(__import__("pathlib").Path(__file__).resolve().parents[1]); R = H + r"/results_vocab"
E = pd.read_csv(H + r"/results_all/exposure.csv"); E["ter"] = pd.qcut(E.code_q.rank(method="first"), 3, labels=["low", "mid", "high"])
M = pd.concat([pd.read_parquet(f, columns=["id", "date", "ntok", "ext_types", "site"]) for f in glob.glob(H + r"/vocab/mk_*.parquet")]); M = M[M.site.isin(E.site)]
M["month"] = M.date.str[:7]; M["ext"] = 100 * M.ext_types / M.ntok.clip(lower=20)
G = M.groupby(["site", "month"]).agg(n=("id", "size"), ext=("ext", "mean")).reset_index(); G = G[(G.n >= 20) & G.month.between("2019-01", "2024-03")].merge(E[["site", "ter"]], on="site")
G["q"] = G.month.str[:4] + "Q" + ((G.month.str[5:7].astype(int) - 1) // 3 + 1).astype(str)
out = {}
for k, g in list(G.groupby("ter", observed=True)) + [("all", G)]:
    s = g.groupby("month").ext.mean(); t = np.arange(len(s)); f = s.index < "2022-12"; b = np.polyfit(t[f], s[f], 1); r = s / np.polyval(b, t)
    out[k] = r.groupby(s.index.str[:4] + "Q" + ((s.index.str[5:7].astype(int) - 1) // 3 + 1).astype(str)).mean()
O = pd.DataFrame(out); O.to_csv(R + r"/tercile_ratio.csv"); print(O.round(3).to_string())
plt.rcParams.update({"font.family": "serif", "font.size": 7}); fig, ax = plt.subplots(figsize=(3.3, 2.1))
x = np.arange(len(O)); sty = {"high": ("#b2182b", "-"), "mid": ("#888888", "--"), "low": ("#2166ac", ":")}
for k in ["high", "mid", "low"]: ax.plot(x, O[k], color=sty[k][0], ls=sty[k][1], lw=1.2, label=f"{k} exposure")
ax.axhline(1, color="k", lw=.4); i0 = list(O.index).index("2022Q4"); ax.axvline(i0 + 2 / 3, color="k", lw=.5, ls="--")
ax.set_xticks(x[::4]); ax.set_xticklabels(O.index[::4]); ax.set_ylabel("marker rate / own pre-trend"); ax.legend(frameon=False, fontsize=6)
ax.spines[["top", "right"]].set_visible(False); plt.tight_layout(); plt.savefig(H + r"/paper/fig_terciles.pdf"); plt.savefig(H + r"/paper/fig_terciles.png", dpi=160)
