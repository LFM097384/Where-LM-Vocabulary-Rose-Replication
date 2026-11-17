"""Survivorship test. Answers created 2019-01..2023-08 that are in the 2023-09 snapshot; 'deleted' = absent from the 2024-04 dump.
1. Deletion rate by month x exposure (event study, rank slope).
2. Marker rate (ext types per 100 tokens), list share, log words of deleted vs surviving answers, by period and exposure.
3. Re-estimate the 2023 dose-response on the snapshot (everything posted, before later deletions) vs the 2024 dump (survivors only),
   window 2019-01..2023-08, so the only difference is the later deletions."""
import glob, os, numpy as np, pandas as pd, statsmodels.formula.api as smf
H = str(__import__("pathlib").Path(__file__).resolve().parents[1]); R = H + r"/results_vocab"
E = pd.read_csv(H + r"/results_all/exposure.csv"); E["rank"] = E.code_q.rank(pct=True) - .5
S = []
for f in glob.glob(H + r"/snap/snap_*.parquet"):
    s = os.path.basename(f)[5:-8]
    if s not in set(E.site) or not os.path.exists(H + rf"/vocab/mk_{s}.parquet"): continue
    a = pd.read_parquet(f); a = a[a.date < "2023-09-01"]
    mk = pd.read_parquet(H + rf"/vocab/mk_{s}.parquet", columns=["id", "ext_types", "ntok"]).rename(columns={"ext_types": "ext24", "ntok": "ntok24"})
    a = a.merge(mk, on="id", how="left"); a["deleted"] = a.ext24.isna().astype(float); S.append(a)
A = pd.concat(S); A = A[A.ntok > 0].merge(E[["site", "rank"]], on="site")
A["month"] = A.date.str[:7]; A["ext"] = 100 * A.ext_types / A.ntok.clip(lower=20); A["lw"] = np.log(A.wc.clip(lower=1)); A["list"] = (A.nli > 0).astype(float)
A["period"] = np.where(A.month >= "2022-12", "post", np.where(A.month >= "2021-12", "pre1", "pre2"))
print("sites", A.site.nunique(), "answers", len(A), "deleted", int(A.deleted.sum()))
# 1. deletion rate panel
P = A.groupby(["site", "month"]).agg(n=("id", "size"), dele=("deleted", "mean"), ext=("ext", "mean"), lw=("lw", "mean"), lst=("list", "mean")).reset_index()
P["t"] = P.month.map(lambda m: (int(m[:4]) - 2022) * 12 + int(m[5:7]) - 12); P = P.merge(E[["site", "rank"]], on="site")
# survivors-only panel for the same answers
Q = A[A.deleted == 0].groupby(["site", "month"]).agg(n=("id", "size"), ext=("ext", "mean"), lw=("lw", "mean"), lst=("list", "mean")).reset_index()
Q["t"] = Q.month.map(lambda m: (int(m[:4]) - 2022) * 12 + int(m[5:7]) - 12); Q = Q.merge(E[["site", "rank"]], on="site")
def fit(z, y, trend):
    z = z.dropna(subset=[y]).copy(); z["post"] = (z.t >= 0).astype(int)
    m = smf.wls(f"{y} ~ C(site) + C(month) + rank:post" + (" + C(site):t" if trend else ""), z, weights=np.minimum(z.n, 2000)).fit(
        cov_type="cluster", cov_kwds={"groups": z.site.astype("category").cat.codes})
    return m.params["rank:post"], m.bse["rank:post"], m.pvalues["rank:post"]
out = []
for lab, d in [("all_posted(snapshot)", P), ("survivors(2024 dump)", Q)]:
    for y in ["ext", "lw", "lst"]:
        for tr in [False, True]:
            b, se, p = fit(d, y, tr); out.append(dict(sample=lab, outcome=y, trend=tr, b=b, se=se, p=p))
for tr in [False, True]:
    b, se, p = fit(P, "dele", tr); out.append(dict(sample="deletion rate", outcome="dele", trend=tr, b=b, se=se, p=p))
EP = pd.read_csv(R + r"/exposure_plus_exante.csv")[["site", "r_edit"]]
for lab, d in [("all_posted(snapshot)", P), ("survivors(2024 dump)", Q), ("deletion rate", P)]:
    z = d.merge(EP, on="site"); z["post"] = (z.t >= 0).astype(int)
    for y in (["dele"] if lab == "deletion rate" else ["ext"]):
        for tr in [False, True]:
            m = smf.wls(f"{y} ~ C(site) + C(month) + rank:post + r_edit:post" + (" + C(site):t" if tr else ""), z, weights=np.minimum(z.n, 2000)).fit(cov_type="cluster", cov_kwds={"groups": z.site.astype("category").cat.codes})
            for k in ["rank", "r_edit"]: out.append(dict(sample=lab + " joint", outcome=y + ":" + k, trend=tr, b=m.params[k + ":post"], se=m.bse[k + ":post"], p=m.pvalues[k + ":post"]))
O = pd.DataFrame(out); O.to_csv(R + r"/survival_dose_exante.csv", index=False)
# 2. deleted vs surviving, by period and exposure half
A["hi"] = A["rank"] > 0
G = A.groupby(["period", "hi", "deleted"]).agg(n=("id", "size"), ext=("ext", "mean"), lw=("lw", "mean"), lst=("list", "mean")).round(4)
G.to_csv(R + r"/survival_contrast_exante.csv")
# answer-level: is marker rate predictive of deletion more after Dec 2022, and more so on high-exposure sites?
A["post"] = (A.period == "post").astype(int); A["extz"] = (A.ext - A.ext.mean()) / A.ext.std()
m = smf.ols("deleted ~ C(site) + C(month) + extz + extz:post + extz:post:rank + lw + list", A.sample(min(len(A), 800_000), random_state=1)).fit(
    cov_type="cluster", cov_kwds={"groups": A.sample(min(len(A), 800_000), random_state=1).site.astype("category").cat.codes})
coef = m.params[["extz", "extz:post", "extz:post:rank"]]; pv = m.pvalues[coef.index]
# deletion event study
P["yr"] = P.month.str[:4] + np.where(P.month.str[5:7].astype(int) <= 6, "H1", "H2"); terms = []
for pp in sorted(P.yr.unique()):
    if pp == "2022H2": continue
    c = "ev_" + pp; P[c] = (P.yr == pp) * P["rank"]; terms.append(c)
me = smf.wls("dele ~ C(site) + C(month) + " + " + ".join(terms), P, weights=np.minimum(P.n, 2000)).fit(cov_type="cluster", cov_kwds={"groups": P.site.astype("category").cat.codes})
ev = pd.DataFrame({"yr": [t[3:] for t in terms], "b": [me.params[t] for t in terms], "se": [me.bse[t] for t in terms]}); ev.to_csv(R + r"/event_deletion_snap_exante.csv", index=False)
traj = A.groupby(["month", "hi"]).deleted.mean().unstack().round(4); traj.to_csv(R + r"/deletion_month_snap_exante.csv")
# revisions between Sep 2023 and Apr 2024 among survivors: did marker counts change in place?
B = A[A.deleted == 0].copy(); B["chg"] = (B.ntok24 != B.ntok).astype(float); B["dext"] = B.ext24 - B.ext_types; B["lost"] = ((B.ext_types > 0) & (B.ext24 < B.ext_types)).astype(float)
B = B.merge(pd.read_csv(R + r"/exposure_plus_exante.csv")[["site", "r_edit"]], on="site"); B["ehi"] = B.r_edit > 0
RV = B.groupby(["period", "ehi"]).agg(n=("id", "size"), changed=("chg", "mean"), dext=("dext", "mean"), lost=("lost", "mean"), ext_snap=("ext_types", "mean"), ext_24=("ext24", "mean"))
RV.to_csv(R + r"/survival_revisions_exante.csv"); print(RV.round(5).to_string())
DL = A.merge(pd.read_csv(R + r"/exposure_plus_exante.csv")[["site", "r_edit"]], on="site"); DL["ehi"] = DL.r_edit > 0; DL["anyx"] = DL.ext_types > 0
print(DL.groupby(["period", "ehi", "anyx"]).deleted.agg(["mean", "size"]).round(4).to_string())
DL.groupby(["period", "ehi", "anyx"]).deleted.agg(["mean", "size"]).to_csv(R + r"/survival_del_by_marker_exante.csv")
pd.set_option("display.width", 200); pd.set_option("display.max_rows", 200)
print(O.round(4).to_string()); print(G.to_string()); print(pd.DataFrame({"b": coef, "p": pv}).round(4)); print(ev.round(4).to_string())
print(traj.loc["2022-06":].to_string())


