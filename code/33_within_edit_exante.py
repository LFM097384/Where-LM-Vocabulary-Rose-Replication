import glob, numpy as np, pandas as pd, statsmodels.formula.api as smf
H = str(__import__("pathlib").Path(__file__).resolve().parents[1]); R = H + r"/results_vocab"
E = pd.read_csv(R + r"/exposure_plus_exante.csv")
M = pd.concat([pd.read_parquet(f, columns=["id", "owner", "date", "ntok", "ext_types", "site"]) for f in glob.glob(H + r"/vocab/mk_*.parquet")]); M = M[M.site.isin(E.site)]
M["month"] = M.date.str[:7]; M["ext"] = 100 * M.ext_types / M.ntok.clip(lower=20); M["anyx"] = (M.ext_types > 0).astype(float)
out = []
for c, lab in [("2022-12", "chatgpt"), ("2021-12", "placebo")]:
    Pm = pd.Period(c, "M"); pre = {str(Pm - k) for k in range(1, 13)}; post = {str(Pm + k) for k in range(12)}
    x = M[M.owner > 0].copy(); x["w"] = np.where(x.month.isin(pre), 0, np.where(x.month.isin(post), 1, -1)); x = x[x.w >= 0]
    g = x.groupby(["site", "owner", "w"]).agg(e=("ext", "mean"), a=("anyx", "mean"), n=("id", "size")).unstack()
    g = g[(g[("n", 0)] >= 3) & (g[("n", 1)] >= 3)]
    d = pd.DataFrame({"de": g[("e", 1)] - g[("e", 0)], "da": g[("a", 1)] - g[("a", 0)]}).reset_index().merge(E[["site", "rank", "r_edit"]], on="site")
    for y in ["de", "da"]:
        for f in ["rank", "r_edit", "rank + r_edit"]:
            m = smf.ols(f"{y} ~ {f}", d).fit(cov_type="cluster", cov_kwds={"groups": d.site.astype("category").cat.codes})
            for k in f.split(" + "): out.append(dict(window=lab, outcome=y, spec=f, term=k, b=m.params[k], p=m.pvalues[k], authors=len(d)))
    # site-month panel of the same answers' share: aggregate effect on panel restricted to those authors
O = pd.DataFrame(out); O.to_csv(R + r"/within_edit_exante.csv", index=False); print(O.round(4).to_string())


