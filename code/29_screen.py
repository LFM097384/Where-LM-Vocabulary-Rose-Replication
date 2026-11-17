"""Site-level screen: y = log change of marker score relative to matched control (2021Q1-22Q3 -> 2023Q2-24Q1); and placebo window.
Univariate WLS on each centred trait rank (15 traits), then joint of code+edited, code+edited+size+len."""
import numpy as np, pandas as pd, statsmodels.formula.api as smf
H = str(__import__("pathlib").Path(__file__).resolve().parents[1]); P = pd.read_csv(H + r"/results_vocab/ddd_panel.csv")
E = pd.read_csv(H + r"/results_all/exposure.csv").set_index("site")
pn = pd.read_csv(H + r"/results_all/panel.csv"); pp = pn[pn.month.between("2021-01", "2022-11")]
T = pp.groupby("site").apply(lambda g: pd.Series({c: np.average(g[c].fillna(g[c].mean()), weights=g.n.fillna(0) + 1e-9) for c in ["list", "head", "edited", "lw_sd", "lqw_m", "heres", "summary", "dv"]}))
T["lnq"] = np.log1p(pp.groupby("site").nq.mean()); T["apq"] = pp.groupby("site").n.sum() / pp.groupby("site").nq.sum()
T = T.join(E[["code_q", "code_a", "math_a", "n_pre", "lw_pre"]]); Tr = T.rank(pct=True) - .5
def per(a, b):
    g = P[P.q.isin(a)].groupby("site").agg(mk=("mk", "mean"), ctl=("ctl", "mean"), n=("n", "sum")); h = P[P.q.isin(b)].groupby("site").agg(mk=("mk", "mean"), ctl=("ctl", "mean"), n=("n", "sum"))
    d = pd.DataFrame({"y": np.log(h.mk / g.mk) - np.log(h.ctl / g.ctl), "w": np.sqrt(h.n + g.n)}).replace([np.inf, -np.inf], np.nan).dropna(); return d.join(Tr)
pre = ["2021Q1", "2021Q2", "2021Q3", "2021Q4", "2022Q1", "2022Q2", "2022Q3"]; post = ["2023Q2", "2023Q3", "2023Q4", "2024Q1"]
plA = ["2019Q1", "2019Q2", "2019Q3", "2019Q4", "2020Q1", "2020Q2", "2020Q3"]; plB = ["2021Q1", "2021Q2", "2021Q3", "2021Q4"]
rows = []
for lab, (a, b) in {"chatgpt": (pre, post), "placebo": (plA, plB)}.items():
    d = per(a, b)
    for c in T.columns:
        m = smf.wls(f"y ~ {c}", d, weights=d.w).fit(cov_type="HC1"); rows.append(dict(window=lab, trait=c, b=m.params[c], p=m.pvalues[c], r2=m.rsquared))
    for f in ["y ~ code_q + edited", "y ~ code_q + edited + n_pre + lw_pre", "y ~ code_q + edited + n_pre + lw_pre + math_a + summary + lqw_m"]:
        m = smf.wls(f, d, weights=d.w).fit(cov_type="HC1"); print(lab, f, {k: (round(m.params[k], 3), round(m.pvalues[k], 3)) for k in m.params.index if k != "Intercept"}, "r2", round(m.rsquared, 3))
O = pd.DataFrame(rows); O.to_csv(H + r"/results_vocab/site_screen.csv", index=False); print(O.round(3).sort_values(["window", "p"]).to_string())
