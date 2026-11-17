"""Dose-response across all SE sites.
Exposure (pre-determined, 2021-01..2022-11): share of questions with a code block (main), and share of answers with code.
Panel: site x month, 2019-01..2024-03. y_sm = monthly mean of answer outcome. Spec:
  y ~ C(site) + C(month) + expo:post [+ C(site):t]   weights = n answers (capped), cluster by site.
Outcomes: log words, SD of log words, list share, header share, phrase rates, log question volume, log question words,
answer log words conditional on question log words. Within-author: change in mean log words for authors active in both windows.
Placebo date: Dec 2021 (sample 2019-01..2022-11).  Also: binned exposure quintile event study for log words.
"""
import os, glob
import numpy as np, pandas as pd, statsmodels.formula.api as smf
B = str(__import__("pathlib").Path(__file__).resolve().parents[1])
R = os.path.join(B, "results_all"); os.makedirs(R, exist_ok=True)
fs = sorted(glob.glob(B + r"/data_all/answers_*.parquet"))
rows, auth, expo = [], [], []
for f in fs:
    s = os.path.basename(f)[8:-8]
    a = pd.read_parquet(f); q = pd.read_parquet(f.replace("answers_", "questions_"))
    if len(a) == 0 or len(q) == 0: continue
    a = a[(a.wc >= 1) & (a.date >= "2019-01-01") & (a.date < "2024-04-01")].copy()
    q = q[(q.qdate >= "2019-01-01") & (q.qdate < "2024-04-01")].copy()
    if len(a) < 500: continue
    a["month"] = a.date.str[:7]; q["month"] = q.qdate.str[:7]; a["lw"] = np.log(a.wc)
    a = a.merge(q[["id", "qwc"]].rename(columns={"id": "parent"}), on="parent", how="left"); a["lqw"] = np.log1p(a.qwc)
    pre = (a.month >= "2021-01") & (a.month <= "2022-11"); qpre = (q.month >= "2021-01") & (q.month <= "2022-11")
    if pre.sum() < 100: continue
    ok = a.lqw.notna() & pre
    b = np.polyfit(a.lqw[ok], a.lw[ok], 1) if ok.sum() > 50 else (0, a.lw[pre].mean())
    a["lw_cq"] = np.where(a.lqw.notna(), a.lw - np.polyval(b, a.lqw.fillna(0)), np.nan)
    a["list"] = (a.nli > 0).astype(float); a["head"] = (a.nhd > 0).astype(float)
    a["heres_any"] = (a.heres > 0).astype(float); a["summary_any"] = (a.summary > 0).astype(float); a["dv"] = 100 * a.delve / a.wc
    expo.append(dict(site=s, code_q=(q.qnpre[qpre] > 0).mean(), code_a=((a.npre + a.ncode)[pre] > 0).mean(),
                     math_a=(a.nmath[pre] > 0).mean(), n_pre=int(pre.sum()), lw_pre=a.lw[pre].mean()))
    g = a.groupby("month").agg(n=("id", "size"), lw=("lw", "mean"), lw_sd=("lw", "std"), lw_cq=("lw_cq", "mean"),
                               list=("list", "mean"), head=("head", "mean"), heres=("heres_any", "mean"),
                               summary=("summary_any", "mean"), dv=("dv", "mean"), edited=("edited", "mean"))
    gu = a[~a.edited].groupby("month").lw.mean().rename("lw_unedited")
    gq = q.groupby("month").agg(nq=("id", "size"), lqw_m=("qwc", lambda v: np.log1p(v).mean()))
    g = g.join(gu).join(gq, how="outer").reset_index(); g["site"] = s; rows.append(g)
    # within author, ChatGPT window and placebo window
    for c, lab in [("2022-12", "chatgpt"), ("2021-12", "placebo")]:
        P = pd.Period(c, "M"); prem = {str(P - k) for k in range(1, 13)}; postm = {str(P + k) for k in range(0, 12)}
        x = a[a.owner > 0]; x = x.assign(w=np.where(x.month.isin(prem), 0, np.where(x.month.isin(postm), 1, -1))); x = x[x.w >= 0]
        if x.w.nunique() < 2: continue
        c2 = x.groupby(["owner", "w"]).lw.agg(["mean", "size"]).unstack()
        c2 = c2[(c2[("size", 0)] >= 3) & (c2[("size", 1)] >= 3)]
        if len(c2): auth.append(dict(site=s, window=lab, n_auth=len(c2), dlw=(c2[("mean", 1)] - c2[("mean", 0)]).mean()))
    print(s, len(a), flush=True)
P_ = pd.concat(rows, ignore_index=True); E = pd.DataFrame(expo); A = pd.DataFrame(auth)
P_ = P_[P_.month.between("2019-01", "2024-03")].copy()
P_["t"] = P_.month.map(lambda m: (int(m[:4]) - 2022) * 12 + int(m[5:7]) - 12)
P_ = P_.merge(E, on="site"); P_["lnq"] = np.log(P_.nq.fillna(0) + 1)
for c in ["code_q", "code_a", "math_a"]: P_[c + "_z"] = (P_[c] - E[c].mean()) / E[c].std()
E["code_rank"] = E.code_q.rank(pct=True); E["top"] = (E.code_rank > .8).astype(int); E["bot"] = (E.code_rank <= .2).astype(int)
P_ = P_.merge(E[["site", "code_rank", "top", "bot"]], on="site")
P_["code_rank_c"] = P_.code_rank - .5
P_.to_csv(R + r"/panel.csv", index=False); E.to_csv(R + r"/exposure.csv", index=False); A.to_csv(R + r"/within_author_sites.csv", index=False)
print("sites", P_.site.nunique(), "site-months", len(P_))

def fit(y, ex="code_q_z", trend=True, placebo=False, d=None):
    z = (P_ if d is None else d).dropna(subset=[y]).copy()
    if placebo: z = z[z.t < 0].copy(); z["post"] = (z.t >= -12).astype(int)
    else: z["post"] = (z.t >= 0).astype(int)
    z["w"] = np.minimum(z.n.fillna(1), 2000)
    f = f"{y} ~ C(site) + C(month) + {ex}:post" + (" + C(site):t" if trend else "")
    m = smf.wls(f, z, weights=z.w).fit(cov_type="cluster", cov_kwds={"groups": z.site.astype("category").cat.codes})
    k = f"{ex}:post"; return m.params[k], m.bse[k], m.pvalues[k], z.site.nunique()
OUT = ["lw", "lw_unedited", "lw_cq", "lw_sd", "list", "head", "heres", "summary", "dv", "lnq", "lqw_m"]
res = []
for y in OUT:
    for ex in ["code_q_z", "code_a_z"]:
        for tr in [True, False]:
            for pl in [False, True]:
                b, se, p, ns = fit(y, ex, tr, pl); res.append(dict(outcome=y, exposure=ex, trend=tr, placebo=pl, b=b, se=se, p=p, sites=ns))
# with math exposure as competing dimension
for y in ["lw", "list", "lnq"]:
    z = P_.dropna(subset=[y]).copy(); z["post"] = (z.t >= 0).astype(int); z["w"] = np.minimum(z.n.fillna(1), 2000)
    m = smf.wls(f"{y} ~ C(site) + C(month) + code_q_z:post + math_a_z:post + C(site):t", z, weights=z.w).fit(
        cov_type="cluster", cov_kwds={"groups": z.site.astype("category").cat.codes})
    for k in ["code_q_z:post", "math_a_z:post"]: res.append(dict(outcome=y, exposure=k, trend=True, placebo=False, b=m.params[k], se=m.bse[k], p=m.pvalues[k], sites=z.site.nunique()))
# rank exposure, full sample and 2021+ sample (no trend), plus top vs bottom quintile
D21 = P_[P_.t >= -23]
TB = P_[(P_.top == 1) | (P_.bot == 1)]
for y in ["lw", "lw_unedited", "lw_cq", "lw_sd", "list", "head", "dv", "heres", "lnq", "lqw_m"]:
    for lab, d, ex, tr in [("rank_full_trend", P_, "code_rank_c", True), ("rank_2021_notrend", D21, "code_rank_c", False),
                           ("top_vs_bot_trend", TB, "top", True), ("top_vs_bot_2021", TB[TB.t >= -23], "top", False)]:
        for pl in [False, True]:
            dd = d if not pl or tr else (P_ if ex == "code_rank_c" else TB)
            if pl and not tr: dd = dd[dd.t >= -35]   # placebo: 2020-01..2022-11, fake break Dec 2021
            b, se, p, ns = fit(y, ex, tr, pl, dd); res.append(dict(outcome=y, exposure=lab, trend=tr, placebo=pl, b=b, se=se, p=p, sites=ns))
RES = pd.DataFrame(res); RES.to_csv(R + r"/dose.csv", index=False)
# event study with CIs: half-year x rank interaction, site + month FE, ref 2022H2
Z2 = P_.dropna(subset=["lw"]).copy(); Z2["yr"] = Z2.month.str[:4] + np.where(Z2.month.str[5:7].astype(int) <= 6, "H1", "H2")
Z2["w"] = np.minimum(Z2.n, 2000); per = sorted(Z2.yr.unique()); ref = "2022H2"; terms = []
for pp in per:
    if pp == ref: continue
    c = "ev_" + pp; Z2[c] = (Z2.yr == pp) * Z2.code_rank_c; terms.append(c)
for y in ["lw", "list", "head"]:
    m = smf.wls(f"{y} ~ C(site) + C(month) + " + " + ".join(terms), Z2.dropna(subset=[y]), weights=Z2.dropna(subset=[y]).w).fit(
        cov_type="cluster", cov_kwds={"groups": Z2.dropna(subset=[y]).site.astype("category").cat.codes})
    ev = pd.DataFrame({"yr": [t[3:] for t in terms], "b": [m.params[t] for t in terms], "se": [m.bse[t] for t in terms]})
    ev = pd.concat([ev, pd.DataFrame({"yr": [ref], "b": [0.0], "se": [0.0]})]).sort_values("yr"); ev.to_csv(R + rf"/event_rank_{y}.csv", index=False)
# quintile event study (log words, relative to 2021-22 site mean, site-trend removed using 2019-01..2022-11)
E["qnt"] = pd.qcut(E.code_q.rank(method="first"), 5, labels=False); Z = P_.merge(E[["site", "qnt"]], on="site")
def detr(g):
    pre = g[g.t < 0]; b = np.polyfit(pre.t, pre.lw, 1); g = g.copy(); g["lw_dt"] = g.lw - np.polyval(b, g.t); return g
Z = Z.dropna(subset=["lw"]).groupby("site", group_keys=False).apply(detr)
Z["yr"] = Z.month.str[:4] + np.where(Z.month.str[5:7].astype(int) <= 6, "H1", "H2")
ES = Z.groupby(["qnt", "yr"]).apply(lambda g: np.average(g.lw_dt, weights=np.minimum(g.n, 2000))).unstack(0); ES.to_csv(R + r"/event_quintile.csv")
# within-author dose
Aw = A.merge(E, on="site"); WA = []
for lab, g in Aw.groupby("window"):
    m = smf.wls("dlw ~ code_q", g, weights=np.sqrt(g.n_auth)).fit(cov_type="HC1")
    WA.append(dict(window=lab, sites=len(g), authors=int(g.n_auth.sum()), slope=m.params["code_q"], p=m.pvalues["code_q"], mean=np.average(g.dlw, weights=g.n_auth)))
WA = pd.DataFrame(WA); WA.to_csv(R + r"/within_author_dose.csv", index=False)
pd.set_option("display.width", 220); pd.set_option("display.max_rows", 300)
print(E.describe().round(3).to_string()); print(RES.round(4).to_string()); print(ES.round(3).to_string()); print(WA.round(4).to_string())
