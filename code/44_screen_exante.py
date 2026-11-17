# 38_screen2 with r_edit replaced by the ex-ante 30-day edit share (r_ea30)
"""Saved site screen: per-site log change marker vs matched control; edit/code single and joint; drop small sites; 2019-20 edit share;
without ensure; blockchain exclusion (9 true blockchain sites); word-level edit gradient (site-level log ratio per word, top words)."""
import os, numpy as np, pandas as pd, statsmodels.formula.api as smf
H = str(__import__("pathlib").Path(__file__).resolve().parents[1]); R = H + r"/results_vocab"; V = H + r"/vocab"
D = pd.read_csv(R + r"/ddd_panel.csv"); D["mk_ne"] = D.y_ne + D.ctl; E = pd.read_csv(R + r"/exposure_plus.csv").set_index("site"); E["r_edit"] = pd.read_csv(R + r"/exante_edit.csv").set_index("site").r_ea30.reindex(E.index)
pn = pd.read_csv(H + r"/results_all/panel.csv")
e19 = pn[pn.month.between("2019-01", "2020-12")].groupby("site").apply(lambda g: np.average(g.edited.fillna(0), weights=g.n.fillna(0) + 1e-9))
pre = ["2021Q1", "2021Q2", "2021Q3", "2021Q4", "2022Q1", "2022Q2", "2022Q3"]; post = ["2023Q2", "2023Q3", "2023Q4", "2024Q1"]
plA = ["2019Q1", "2019Q2", "2019Q3", "2019Q4", "2020Q1", "2020Q2", "2020Q3"]; plB = ["2021Q1", "2021Q2", "2021Q3", "2021Q4"]
BC = ["ethereum", "bitcoin", "eosio", "tezos", "solana", "cardano", "monero", "stellar", "substrate"]
rows = []
def per(a, b):
    g = D[D.q.isin(a)].groupby("site")[["mk", "mk_ne", "ctl", "n"]].agg({"mk": "mean", "mk_ne": "mean", "ctl": "mean", "n": "sum"})
    h = D[D.q.isin(b)].groupby("site")[["mk", "mk_ne", "ctl", "n"]].agg({"mk": "mean", "mk_ne": "mean", "ctl": "mean", "n": "sum"})
    d = pd.DataFrame({"y": np.log(h.mk / g.mk) - np.log(h.ctl / g.ctl), "yne": np.log(h.mk_ne / g.mk_ne) - np.log(h.ctl / g.ctl), "w": np.sqrt(h.n + g.n), "npost": h.n}).replace([np.inf, -np.inf], np.nan).dropna()
    d = d.join(E[["rank", "r_edit", "r_size", "r_len", "edited"]]); d["e19r"] = (e19.reindex(d.index).rank(pct=True) - .5); return d
for lab, (a, b) in {"chatgpt": (pre, post), "placebo": (plA, plB)}.items():
    d = per(a, b)
    samples = {"all": d, "drop_small(post n>=1000)": d[d.npost >= 1000], "drop_blockchain9": d[~d.index.isin(BC)]}
    for sl, z in samples.items():
        for y in ["y", "yne"]:
            for f in ["r_edit", "rank", "rank + r_edit", "e19r", "rank + r_edit + r_size + r_len"]:
                m = smf.wls(f"{y} ~ {f}", z, weights=z.w).fit(cov_type="HC1")
                for k in f.split(" + "): rows.append(dict(window=lab, sample=sl, outcome=y, spec=f, term=k, b=m.params[k], p=m.pvalues[k], r2=m.rsquared, sites=len(z)))
    if lab == "chatgpt":
        pass
        print("top rises", d.sort_values("y", ascending=False).head(8)[["y", "edited"]].round(2).to_dict())
        print("sizes", d.npost.describe().round(0).to_dict())
O = pd.DataFrame(rows); O.to_csv(R + r"/site_screen_exante.csv", index=False); pd.set_option("display.width", 220); pd.set_option("display.max_rows", 400)
print(O[(O.outcome == "y") | (O.spec.isin(["r_edit", "rank + r_edit"]))].round(3).to_string())
# word-level: per-word site-level log change of document share (pooled pre vs post), regressed on r_edit; words with enough mass
src = open(H + r"/code/07_vocab.py", encoding="utf8").read(); i = src.index("EXT = set("); j = src.index(".split())", i) + 9; EXT = sorted(eval(src[i + 6:j]))
vs, ns = [], []
for s in E.index:
    f = V + rf"/vocab_{s}.parquet"
    if not os.path.exists(f): continue
    v = pd.read_parquet(f); v = v[v.word.isin(EXT) & v.q.isin(pre + post)]; v["site"] = s; vs.append(v)
    n = pd.read_parquet(V + rf"/nq_{s}.parquet"); n = n[n.q.isin(pre + post)]; n["site"] = s; ns.append(n)
Vw = pd.concat(vs); N = pd.concat(ns); Vw["per"] = np.where(Vw.q.isin(post), "post", "pre"); N["per"] = np.where(N.q.isin(post), "post", "pre")
dfw = Vw.groupby(["site", "word", "per"]).df.sum().unstack(fill_value=0); nn = N.groupby(["site", "per"]).n.sum().unstack()
wr = []
for w in EXT:
    if w not in dfw.index.get_level_values("word"): continue
    x = dfw.xs(w, level="word").reindex(nn.index, fill_value=0)
    sh = (x + .5).div(nn + 1, axis=0); y = np.log(sh.post / sh.pre); z = pd.DataFrame({"y": y, "w": np.sqrt(nn.sum(axis=1))}).join(E[["r_edit", "rank"]]).dropna()
    if (x.sum(axis=1) > 0).sum() < 60: continue
    m = smf.wls("y ~ r_edit", z, weights=z.w).fit(cov_type="HC1"); wr.append(dict(word=w, b_edit=m.params["r_edit"], p=m.pvalues["r_edit"], tot=int(x.values.sum())))
WR = pd.DataFrame(wr).sort_values("b_edit"); WR.to_csv(R + r"/word_edit_gradient_exante.csv", index=False); print(WR.round(3).to_string())
print("words negative", (WR.b_edit < 0).sum(), "of", len(WR), "; sig neg", ((WR.b_edit < 0) & (WR.p < .05)).sum(), "; sig pos", ((WR.b_edit > 0) & (WR.p < .05)).sum())

