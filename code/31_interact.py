import numpy as np, pandas as pd, statsmodels.formula.api as smf
H = str(__import__("pathlib").Path(__file__).resolve().parents[1]); R = H + r"/results_vocab"
P = pd.read_parquet(R + r"/panel_edit.parquet")
out = []
for y in ["ext", "anyx", "list"]:
    for tr in [False, True]:
        for pl in [False, True]:
            z = P.dropna(subset=[y]).copy()
            if pl: z = z[z.t < 0]; z["post"] = (z.t >= -12).astype(int)
            z["ce"] = z["rank"] * z.r_edit
            m = smf.wls(f"{y} ~ C(site) + C(month) + rank:post + r_edit:post + ce:post" + (" + C(site):t" if tr else ""), z, weights=np.minimum(z.n, 2000)).fit(cov_type="cluster", cov_kwds={"groups": z.site.astype("category").cat.codes})
            for k in ["rank", "r_edit", "ce"]: out.append(dict(outcome=y, trend=tr, placebo=pl, term=k, b=m.params[k + ":post"], p=m.pvalues[k + ":post"]))
O = pd.DataFrame(out); O.to_csv(R + r"/code_edit_interaction.csv", index=False); print(O.round(4).to_string())
# 2x2 cells: site log change (marker vs matched control), from ddd panel
D = pd.read_csv(R + r"/ddd_panel.csv"); pre = ["2021Q1", "2021Q2", "2021Q3", "2021Q4", "2022Q1", "2022Q2", "2022Q3"]; post = ["2023Q2", "2023Q3", "2023Q4", "2024Q1"]
g = D[D.q.isin(pre)].groupby("site").agg(mk=("mk", "mean"), ctl=("ctl", "mean"), n=("n", "sum")); h = D[D.q.isin(post)].groupby("site").agg(mk=("mk", "mean"), ctl=("ctl", "mean"), n=("n", "sum"))
d = pd.DataFrame({"y": np.log(h.mk / g.mk) - np.log(h.ctl / g.ctl), "w": np.sqrt(h.n + g.n)}).replace([np.inf, -np.inf], np.nan).dropna().join(D.drop_duplicates("site").set_index("site")[["rank", "r_edit"]])
d["code"] = np.where(d["rank"] > 0, "high code", "low code"); d["edit"] = np.where(d.r_edit > 0, "high edit", "low edit")
c = d.groupby(["code", "edit"]).apply(lambda x: pd.Series({"y": np.average(x.y, weights=x.w), "sites": len(x)})); print(c.round(3)); c.to_csv(R + r"/cells_code_edit.csv")
