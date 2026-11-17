"""Mechanical check: does the edit gradient survive among answers never edited (text as first posted)?
Join answer-level edited flag from data_all/answers_<site>.parquet to vocab/mk_<site>.parquet; build site-month marker panel for
unedited-only and edited-only answers; edit-rank and code-rank dose models; placebo."""
import glob, os, numpy as np, pandas as pd, statsmodels.formula.api as smf
H = str(__import__("pathlib").Path(__file__).resolve().parents[1]); R = H + r"/results_vocab"
E = pd.read_csv(R + r"/exposure_plus.csv"); rows = []
for s in E.site:
    f = H + rf"/data_all/answers_{s}.parquet"; g = H + rf"/vocab/mk_{s}.parquet"
    if not (os.path.exists(f) and os.path.exists(g)): continue
    a = pd.read_parquet(f, columns=["id", "edited"]); m = pd.read_parquet(g, columns=["id", "date", "ntok", "ext_types"]).merge(a, on="id")
    m["month"] = m.date.str[:7]; m = m[m.month.between("2019-01", "2024-03")]; m["ext"] = 100 * m.ext_types / m.ntok.clip(lower=20); m["anyx"] = (m.ext_types > 0).astype(float)
    x = m.groupby(["month", "edited"]).agg(n=("id", "size"), ext=("ext", "mean"), anyx=("anyx", "mean")).reset_index(); x["site"] = s; rows.append(x)
    t = m.groupby("month").edited.mean().reset_index(); t["site"] = s; t["edited_flag"] = "share"; rows.append(t.rename(columns={"edited": "eshare"}))
X = pd.concat(rows); X.to_parquet(R + r"/panel_by_edited.parquet")
S = X[X.edited_flag == "share"][["site", "month", "eshare"]]; X = X[X.edited_flag != "share"].drop(columns=["edited_flag", "eshare"])
X["t"] = X.month.map(lambda m: (int(m[:4]) - 2022) * 12 + int(m[5:7]) - 12); X = X.merge(E[["site", "rank", "r_edit"]], on="site"); X["post"] = (X.t >= 0).astype(int)
out = []
for lab, z0 in [("unedited", X[X.edited == False]), ("edited", X[X.edited == True])]:
    for y in ["ext", "anyx"]:
        for rhs in ["r_edit:post", "rank:post", "rank:post + r_edit:post"]:
            for tr in [False, True]:
                for pl in [False, True]:
                    z = z0.copy()
                    if pl: z = z[z.t < 0]; z["post"] = (z.t >= -12).astype(int)
                    m = smf.wls(f"{y} ~ C(site) + C(month) + {rhs}" + (" + C(site):t" if tr else ""), z, weights=np.minimum(z.n, 2000)).fit(cov_type="cluster", cov_kwds={"groups": z.site.astype("category").cat.codes})
                    for k in [k for k in m.params.index if k.endswith(":post")]: out.append(dict(sample=lab, outcome=y, spec=rhs, trend=tr, placebo=pl, term=k[:-5], b=m.params[k], p=m.pvalues[k]))
O = pd.DataFrame(out); O.to_csv(R + r"/unedited_dose.csv", index=False); pd.set_option("display.width", 200); pd.set_option("display.max_rows", 300)
print(O[O.outcome == "ext"].round(4).to_string())
# edit share over time by edit tercile (do post-2022 answers get edited less on low-edit sites? censoring)
S = S.merge(E[["site", "r_edit"]], on="site"); S["ter"] = pd.cut(S.r_edit, [-1, -1 / 6, 1 / 6, 1], labels=["low", "mid", "high"]); S["yr"] = S.month.str[:4]
print(S.groupby(["yr", "ter"], observed=True).eshare.mean().unstack().round(3))
# marker rate of edited vs unedited answers pre/post
X["per"] = np.where(X.t >= 0, "post", "pre"); print(X.groupby(["per", "edited"]).apply(lambda g: np.average(g.ext, weights=g.n)).round(4))
