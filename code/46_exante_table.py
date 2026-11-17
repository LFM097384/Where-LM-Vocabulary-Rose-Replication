import numpy as np, pandas as pd, statsmodels.formula.api as smf
R = str(__import__("pathlib").Path(__file__).resolve().parents[1]) + "/results_vocab"
S = pd.read_csv(R + r"/exante_edit.csv").set_index("site"); P = pd.read_parquet(R + r"/panel_edit.parquet").merge(S[["r_ea30"]].reset_index(), on="site")
out = []
for y in ["ext", "anyx"]:
    for tr in [False, True]:
        for pl in [False, True]:
            z = P.dropna(subset=[y]).copy()
            if pl: z = z[z.t < 0]; z["post"] = (z.t >= -12).astype(int)
            for rhs in ["r_ea30:post", "rank:post + r_ea30:post", "rank:post + r_ea30:post + r_size:post + r_len:post + r_math:post"]:
                m = smf.wls(f"{y} ~ C(site) + C(month) + {rhs}" + (" + C(site):t" if tr else ""), z, weights=np.minimum(z.n, 2000)).fit(cov_type="cluster", cov_kwds={"groups": z.site.astype("category").cat.codes})
                for k in [k for k in m.params.index if k.endswith(":post")]: out.append(dict(y=y, trend=tr, placebo=pl, spec=rhs.count("+"), term=k[:-5], b=m.params[k], p=m.pvalues[k], sites=z.site.nunique()))
O = pd.DataFrame(out); O.to_csv(R + r"/exante_table.csv", index=False); pd.set_option("display.width", 200); pd.set_option("display.max_rows", 200)
print(O[O.term.isin(["rank", "r_ea30"])].round(4).to_string())
