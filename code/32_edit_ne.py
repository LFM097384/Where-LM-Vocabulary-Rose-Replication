"""Site log change vs matched control, marker list with and without 'ensure': regress on code rank and edit rank."""
import numpy as np, pandas as pd, statsmodels.formula.api as smf
H = str(__import__("pathlib").Path(__file__).resolve().parents[1]); D = pd.read_csv(H + r"/results_vocab/ddd_panel.csv")
D["mk_ne"] = D.y_ne + D.ctl
pre = ["2021Q1", "2021Q2", "2021Q3", "2021Q4", "2022Q1", "2022Q2", "2022Q3"]; post = ["2023Q2", "2023Q3", "2023Q4", "2024Q1"]
plA = ["2019Q1", "2019Q2", "2019Q3", "2019Q4", "2020Q1", "2020Q2", "2020Q3"]; plB = ["2021Q1", "2021Q2", "2021Q3", "2021Q4"]
for lab, (a, b) in {"chatgpt": (pre, post), "placebo": (plA, plB)}.items():
    g = D[D.q.isin(a)].groupby("site")[["mk", "mk_ne", "ctl", "n"]].mean(); h = D[D.q.isin(b)].groupby("site")[["mk", "mk_ne", "ctl", "n"]].mean()
    d = pd.DataFrame({"y": np.log(h.mk / g.mk) - np.log(h.ctl / g.ctl), "yne": np.log(h.mk_ne / g.mk_ne) - np.log(h.ctl / g.ctl), "w": np.sqrt(h.n + g.n)}).replace([np.inf, -np.inf], np.nan).dropna()
    d = d.join(D.drop_duplicates("site").set_index("site")[["rank", "r_edit", "r_size", "r_len"]])
    for y in ["y", "yne"]:
        for f in ["rank", "r_edit", "rank + r_edit", "rank + r_edit + r_size + r_len"]:
            m = smf.wls(f"{y} ~ {f}", d, weights=d.w).fit(cov_type="HC1"); print(lab, y, f, {k: (round(m.params[k], 3), round(m.pvalues[k], 3)) for k in m.params.index if k != "Intercept"})
