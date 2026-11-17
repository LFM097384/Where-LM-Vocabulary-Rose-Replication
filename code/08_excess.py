"""Excess vocabulary and dose-response, with sample splitting.
1. Random split of sites into a discovery half (A) and a test half (B), stratified by code-exposure tercile (seed 2026).
2. On A only: pooled df share per quarter. Expected 2023Q2-2024Q1 share = linear extrapolation of 2021Q1-2022Q3.
   Excess ratio r = observed / max(expected, floor). Excess words: df share >= 1e-4 in both periods, r >= 1.5, top 100 by
   excess gap (obs - exp), with code/tech tokens kept (no filtering beyond length >= 4).
   Placebo list: same procedure, shifted back two years (expected 2021Q2-2022Q1 from 2019Q1-2020Q3).
3. On B: answer-level score = share of distinct excess words present. Site-month panel, dose-response on code exposure rank,
   site + month FE, cluster by site; placebo date Dec 2021. Within author. Swap A and B and report both.
Also scores the external fixed list (ext_types) on all sites."""
import os, glob, re
import numpy as np, pandas as pd, statsmodels.formula.api as smf
B_ = str(__import__("pathlib").Path(__file__).resolve().parents[1]); V = os.path.join(B_, "vocab"); R = os.path.join(B_, "results_vocab"); os.makedirs(R, exist_ok=True)
E = pd.read_csv(B_ + r"/results_all/exposure.csv")
sites = sorted(set(os.path.basename(f)[3:-8] for f in glob.glob(V + r"/mk_*.parquet")) & set(E.site))
E = E[E.site.isin(sites)].copy(); E["rank"] = E.code_q.rank(pct=True) - .5; E["terc"] = pd.qcut(E.code_q.rank(method="first"), 3, labels=False)
rng = np.random.default_rng(2026); E["half"] = 0
for t, g in E.groupby("terc"):
    idx = rng.permutation(g.index); E.loc[idx[: len(idx) // 2], "half"] = 1
print("sites", len(E), E.half.value_counts().to_dict())
QS = [f"{y}Q{q}" for y in range(2019, 2025) for q in range(1, 5) if f"{y}Q{q}" <= "2024Q1"]
qi = {q: i for i, q in enumerate(QS)}
def pooled(ss, fit_q, obs_q):
    """Equal-weight mean over sites of per-site df share, plus share of sites in which the word rose obs vs fit."""
    acc = None; ups = None
    for s in ss:
        v = pd.read_parquet(V + rf"/vocab_{s}.parquet"); n = pd.read_parquet(V + rf"/nq_{s}.parquet").set_index("q").n.reindex(QS)
        D = v.pivot_table(index="word", columns="q", values="df", aggfunc="sum", fill_value=0).reindex(columns=QS, fill_value=0)
        F = D.div(n, axis=1).fillna(0); pres = F[fit_q].mean(axis=1) > 0; up = (F[obs_q].mean(axis=1) > F[fit_q].mean(axis=1)).astype(float).where(pres); pr = pres.astype(float)
        acc = F if acc is None else acc.add(F, fill_value=0); ups = up.fillna(0) if ups is None else ups.add(up.fillna(0), fill_value=0); npres = pr if s == ss[0] else npres.add(pr, fill_value=0)
    F = acc / len(ss); F["_up"] = (ups / npres.clip(lower=1)).reindex(F.index).fillna(0); F["_nsite"] = npres.reindex(F.index).fillna(0); return F
def excess(F, fit_q, obs_q, k=100):
    x = np.array([qi[q] for q in fit_q]); X = np.vstack([x, np.ones_like(x)]).T
    coef = np.linalg.lstsq(X, F[fit_q].T.values, rcond=None)[0]
    xo = np.array([qi[q] for q in obs_q]).mean(); exp = np.maximum(coef[0] * xo + coef[1], 0)
    F = F.copy(); FF = F[fit_q + [q for q in obs_q if q not in fit_q]]; obs = F[obs_q].mean(axis=1).values; pre = F[fit_q].mean(axis=1).values
    d = pd.DataFrame({"word": F.index, "obs": obs, "exp": exp, "pre": pre, "up": F["_up"].values, "nsite": F["_nsite"].values}); d = d[(d.up >= .6) & (d.nsite >= 10)]; d["r"] = d.obs / np.maximum(d.exp, 1e-4); d["gap"] = d.obs - d.exp
    d = d[~d.word.str.contains("gpt|openai|chatbot|llm|copilot|bard|gemini|claude")]
    d = d[(d.word.str.len() >= 4) & (d.pre >= 1e-4) & (d.obs >= 1e-4) & (d.r >= 1.5)].sort_values("gap", ascending=False)
    return d.head(k)
FIT = [q for q in QS if "2021Q1" <= q <= "2022Q3"]; OBS = [q for q in QS if "2023Q2" <= q <= "2024Q1"]
FITp = [q for q in QS if "2019Q1" <= q <= "2020Q3"]; OBSp = [q for q in QS if "2021Q2" <= q <= "2022Q1"]
TOK = None
def fit(z, y, placebo=False, trend=False):
    z = z.dropna(subset=[y]).copy()
    if placebo: z = z[z.t < 0]; z["post"] = (z.t >= -12).astype(int)
    else: z["post"] = (z.t >= 0).astype(int)
    f = f"{y} ~ C(site) + C(month) + rank:post" + (" + C(site):t" if trend else "")
    m = smf.wls(f, z, weights=np.minimum(z.n, 2000)).fit(cov_type="cluster", cov_kwds={"groups": z.site.astype("category").cat.codes})
    return m.params["rank:post"], m.pvalues["rank:post"]
def answer_scores(ss, words, tag):
    """Re-score answers needs per-answer word sets; approximate at site-quarter level from df: score = mean over words of df share.
    Site-month needs answer-level, so we use site-quarter panel here."""
    rows = []
    for s in ss:
        v = pd.read_parquet(V + rf"/vocab_{s}.parquet"); n = pd.read_parquet(V + rf"/nq_{s}.parquet").set_index("q").n
        v = v[v.word.isin(words)]
        sh = (v.groupby("q").df.sum().reindex(n.index, fill_value=0) / n) / len(words)
        for q, val in sh.items(): rows.append(dict(site=s, q=q, score=val * 100, n=n[q]))
    P = pd.DataFrame(rows); P["t"] = P.q.map(lambda q: (int(q[:4]) - 2022) * 12 + (int(q[5]) - 1) * 3 + 1 - 12)  # first month of quarter, Dec 2022 = 0
    P["month"] = P.q; return P.merge(E[["site", "rank"]], on="site")
out, lists = [], {}
for disc in [1, 0]:
    A = E[E.half == disc].site.tolist(); Bt = E[E.half != disc].site.tolist()
    ex = excess(pooled(A, FIT, OBS), FIT, OBS); exp_ = excess(pooled(A, FITp, OBSp), FITp, OBSp)
    lists[f"disc{disc}"] = ex; lists[f"disc{disc}_placebo"] = exp_
    for lab, w in [("excess", ex.word.tolist()), ("excess_placebo_list", exp_.word.tolist())]:
        P = answer_scores(Bt, set(w), lab)
        for pl in [False, True]:
            for tr in [False, True]:
                b, p = fit(P.rename(columns={}), "score", pl, tr)
                out.append(dict(discovery=disc, words=lab, placebo=pl, trend=tr, b=b, p=p, sites=P.site.nunique()))
        P.to_csv(R + rf"/panel_{lab}_test{1-disc}.csv", index=False)
for k, d in lists.items(): d.to_csv(R + rf"/words_{k}.csv", index=False)
O = pd.DataFrame(out); O.to_csv(R + r"/dose_excess.csv", index=False)
# external fixed list, answer level, site-month, plus within-author
M = pd.concat([pd.read_parquet(f) for f in glob.glob(V + r"/mk_*.parquet")]); M = M[M.site.isin(E.site)]
M["month"] = M.date.str[:7]; M["ext"] = 100 * M.ext_types / M.ntok.clip(lower=20); M["any"] = (M.ext_types > 0).astype(float)
P2 = M.groupby(["site", "month"]).agg(n=("id", "size"), ext=("ext", "mean"), anyx=("any", "mean")).reset_index()
P2["t"] = P2.month.map(lambda m: (int(m[:4]) - 2022) * 12 + int(m[5:7]) - 12); P2 = P2.merge(E[["site", "rank"]], on="site")
ext_out = []
for y in ["ext", "anyx"]:
    for pl in [False, True]:
        for tr in [False, True]:
            b, p = fit(P2, y, pl, tr); ext_out.append(dict(outcome=y, placebo=pl, trend=tr, b=b, p=p))
# within author external list
W = []
for c, lab in [("2022-12", "chatgpt"), ("2021-12", "placebo")]:
    Pm = pd.Period(c, "M"); pre = {str(Pm - k) for k in range(1, 13)}; post = {str(Pm + k) for k in range(12)}
    x = M[M.owner > 0].copy(); x["w"] = np.where(x.month.isin(pre), 0, np.where(x.month.isin(post), 1, -1)); x = x[x.w >= 0]
    g = x.groupby(["site", "owner", "w"]).agg(e=("ext", "mean"), n=("id", "size")).unstack()
    g = g[(g[("n", 0)] >= 3) & (g[("n", 1)] >= 3)]; d = (g[("e", 1)] - g[("e", 0)]).rename("d").reset_index().merge(E[["site", "rank"]], on="site")
    m = smf.ols("d ~ rank", d).fit(cov_type="cluster", cov_kwds={"groups": d.site.astype("category").cat.codes})
    W.append(dict(window=lab, authors=len(d), slope=m.params["rank"], p=m.pvalues["rank"], mean=d.d.mean()))
EO = pd.DataFrame(ext_out); EO.to_csv(R + r"/dose_external.csv", index=False); WD = pd.DataFrame(W); WD.to_csv(R + r"/within_external.csv", index=False)
# event study, external list
P2["yr"] = P2.month.str[:4] + np.where(P2.month.str[5:7].astype(int) <= 6, "H1", "H2"); terms = []
for pp in sorted(P2.yr.unique()):
    if pp == "2022H2": continue
    c = "ev_" + pp; P2[c] = (P2.yr == pp) * P2["rank"]; terms.append(c)
m = smf.wls("ext ~ C(site) + C(month) + " + " + ".join(terms), P2, weights=np.minimum(P2.n, 2000)).fit(cov_type="cluster", cov_kwds={"groups": P2.site.astype("category").cat.codes})
ev = pd.DataFrame({"yr": [t[3:] for t in terms], "b": [m.params[t] for t in terms], "se": [m.bse[t] for t in terms]})
ev = pd.concat([ev, pd.DataFrame({"yr": ["2022H2"], "b": [0.0], "se": [0.0]})]).sort_values("yr"); ev.to_csv(R + r"/event_external.csv", index=False)
# per-word pooled trajectory for the external list (all sites, low vs high exposure halves)
pd.set_option("display.width", 220)
for k in ["disc1", "disc0"]: print(k, ", ".join(lists[k].word.head(40)))
for k in ["disc1_placebo", "disc0_placebo"]: print(k, ", ".join(lists[k].word.head(25)))
print(O.round(4).to_string()); print(EO.round(4).to_string()); print(WD.round(4).to_string()); print(ev.round(4).to_string())
