"""Deletion rate from Post-Id gaps. Questions and answers share one Id sequence per site; Ids absent from the dump were
deleted (or are rare tag-wiki posts, absorbed by site FE). Gap after each surviving post is assigned to that post's month.
del_share = missing / (missing + surviving), per site-month, 2019-01..2024-03. Dose-response on code-exposure rank."""
import glob, os, numpy as np, pandas as pd, statsmodels.formula.api as smf
B = str(__import__("pathlib").Path(__file__).resolve().parents[1]); R = B + r"/results_vocab"
E = pd.read_csv(B + r"/results_all/exposure.csv"); E["rank"] = E.code_q.rank(pct=True) - .5
rows = []
for f in glob.glob(B + r"/data_all/answers_*.parquet"):
    s = os.path.basename(f)[8:-8]
    if s not in set(E.site): continue
    a = pd.read_parquet(f, columns=["id", "date"]); q = pd.read_parquet(f.replace("answers_", "questions_"), columns=["id", "qdate"]).rename(columns={"qdate": "date"})
    a["kind"] = "a"; q["kind"] = "q"; x = pd.concat([a, q]).sort_values("id")
    x["gap"] = (x.id.shift(-1) - x.id - 1).clip(lower=0)
    x = x[(x.date >= "2019-01-01") & (x.date < "2024-04-01")].dropna(subset=["gap"]); x["month"] = x.date.str[:7]
    g = x.groupby("month").agg(n=("id", "size"), miss=("gap", "sum"), na=("kind", lambda k: (k == "a").sum())).reset_index()
    g["site"] = s; rows.append(g)
P = pd.concat(rows); P["del_share"] = P.miss / (P.miss + P.n); P["ldel"] = np.log1p(P.miss) - np.log(P.n)
P["t"] = P.month.map(lambda m: (int(m[:4]) - 2022) * 12 + int(m[5:7]) - 12); P = P.merge(E[["site", "rank"]], on="site")
P.to_csv(R + r"/deletion_panel.csv", index=False)
def fit(z, y, placebo=False, trend=False):
    z = z.dropna(subset=[y]).copy()
    if placebo: z = z[z.t < 0]; z["post"] = (z.t >= -12).astype(int)
    else: z["post"] = (z.t >= 0).astype(int)
    f = f"{y} ~ C(site) + C(month) + rank:post" + (" + C(site):t" if trend else "")
    m = smf.wls(f, z, weights=np.minimum(z.n, 2000)).fit(cov_type="cluster", cov_kwds={"groups": z.site.astype("category").cat.codes})
    return m.params["rank:post"], m.pvalues["rank:post"]
out = []
for y in ["del_share"]:
    for pl in [False, True]:
        for tr in [False, True]:
            b, p = fit(P, y, pl, tr); out.append(dict(outcome=y, placebo=pl, trend=tr, b=b, p=p))
O = pd.DataFrame(out); O.to_csv(R + r"/dose_deletion.csv", index=False)
# pooled trajectory by exposure half (unweighted site means)
P["hi"] = P["rank"] > 0; P["yr"] = P.month.str[:4] + np.where(P.month.str[5:7].astype(int) <= 6, "H1", "H2")
T = P.groupby(["yr", "hi"]).del_share.mean().unstack(); T.to_csv(R + r"/deletion_traj.csv")
# event study
terms = []
for pp in sorted(P.yr.unique()):
    if pp == "2022H2": continue
    c = "ev_" + pp; P[c] = (P.yr == pp) * P["rank"]; terms.append(c)
m = smf.wls("del_share ~ C(site) + C(month) + " + " + ".join(terms), P, weights=np.minimum(P.n, 2000)).fit(cov_type="cluster", cov_kwds={"groups": P.site.astype("category").cat.codes})
ev = pd.DataFrame({"yr": [t[3:] for t in terms], "b": [m.params[t] for t in terms], "se": [m.bse[t] for t in terms]}); ev.to_csv(R + r"/event_deletion.csv", index=False)
pd.set_option("display.width", 200); print(O.round(4)); print(T.round(3)); print(ev.round(4))
