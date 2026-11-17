import numpy as np, pandas as pd, statsmodels.formula.api as smf
H = str(__import__("pathlib").Path(__file__).resolve().parents[1]); P = pd.read_csv(H + r"/results_vocab/ddd_panel.csv")
def per(a, b):
    g = P[P.q.isin(a)].groupby("site").agg(mk=("mk", "mean"), ctl=("ctl", "mean"), n=("n", "sum")); h = P[P.q.isin(b)].groupby("site").agg(mk=("mk", "mean"), ctl=("ctl", "mean"), n=("n", "sum"))
    d = pd.DataFrame({"dmk": np.log(h.mk / g.mk), "dctl": np.log(h.ctl / g.ctl), "n": h.n + g.n}).dropna(); d = d[np.isfinite(d.dmk)]
    d["y"] = d.dmk - d.dctl; return d.join(P.drop_duplicates("site").set_index("site")[["rank", "r_size", "r_len", "r_math", "r_edit"]])
pre = ["2021Q1", "2021Q2", "2021Q3", "2021Q4", "2022Q1", "2022Q2", "2022Q3"]; post = ["2023Q2", "2023Q3", "2023Q4", "2024Q1"]
plA = ["2019Q1", "2019Q2", "2019Q3", "2019Q4", "2020Q1", "2020Q2", "2020Q3"]; plB = ["2021Q1", "2021Q2", "2021Q3", "2021Q4"]
for lab, (a, b) in {"chatgpt": (pre, post), "placebo": (plA, plB)}.items():
    d = per(a, b); w = np.sqrt(d.n)
    print(lab, "sites", len(d), "mean y", round(np.average(d.y, weights=w), 4), "mean dmk", round(np.average(d.dmk, weights=w), 4), "mean dctl", round(np.average(d.dctl, weights=w), 4))
    for f in ["y ~ rank", "y ~ r_len", "y ~ r_size", "y ~ rank + r_len + r_size + r_math + r_edit", "dmk ~ rank + r_len + r_size + r_math + r_edit"]:
        m = smf.wls(f, d, weights=w).fit(cov_type="HC1"); print("   ", f, {k: (round(m.params[k], 3), round(m.pvalues[k], 3)) for k in m.params.index if k != "Intercept"})
    d["ter"] = pd.qcut(d["rank"], 3, labels=["low", "mid", "high"]); print(d.groupby("ter", observed=True).apply(lambda g: pd.Series({"y": np.average(g.y, weights=np.sqrt(g.n)), "dmk": np.average(g.dmk, weights=np.sqrt(g.n)), "dctl": np.average(g.dctl, weights=np.sqrt(g.n))})).round(3))
