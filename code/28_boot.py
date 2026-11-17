"""Site-level log change of marker score relative to matched control vocabulary; bootstrap over sites (2000) for mean, tercile means,
high-minus-low; placebo windows. Also edited-share terciles."""
import numpy as np, pandas as pd
H = str(__import__("pathlib").Path(__file__).resolve().parents[1]); P = pd.read_csv(H + r"/results_vocab/ddd_panel.csv")
pre = [f"{y}Q{k}" for y in (2021, 2022) for k in range(1, 5)][:7]; post = ["2023Q2", "2023Q3", "2023Q4", "2024Q1"]
plA = [f"{y}Q{k}" for y in (2019, 2020) for k in range(1, 5)][:7]; plB = ["2021Q1", "2021Q2", "2021Q3", "2021Q4"]
S = P.drop_duplicates("site").set_index("site")[["rank", "r_edit", "r_len", "r_size"]]
def per(a, b):
    g = P[P.q.isin(a)].groupby("site")[["mk", "ctl", "n"]].agg({"mk": "mean", "ctl": "mean", "n": "sum"}); h = P[P.q.isin(b)].groupby("site")[["mk", "ctl", "n"]].agg({"mk": "mean", "ctl": "mean", "n": "sum"})
    d = pd.DataFrame({"dmk": np.log(h.mk / g.mk), "dctl": np.log(h.ctl / g.ctl), "w": np.sqrt(h.n + g.n)}).replace([np.inf, -np.inf], np.nan).dropna()
    d["y"] = d.dmk - d.dctl; return d.join(S)
rng = np.random.default_rng(1); out = []
for lab, (a, b) in {"chatgpt": (pre, post), "placebo": (plA, plB)}.items():
    d = per(a, b); d["ter"] = pd.qcut(d["rank"], 3, labels=["low", "mid", "high"]).astype(str); d["eter"] = pd.qcut(d["r_edit"], 3, labels=["elow", "emid", "ehigh"]).astype(str)
    def stats(x):
        f = lambda m: np.average(x.y[m], weights=x.w[m]); r = {"all": f(np.ones(len(x), bool)), "dmk": np.average(x.dmk, weights=x.w), "dctl": np.average(x.dctl, weights=x.w)}
        for t in ["low", "mid", "high"]: r[t] = f(x.ter == t)
        for t in ["elow", "ehigh"]: r[t] = f(x.eter == t)
        r["high-low"] = r["high"] - r["low"]; r["ehigh-elow"] = r["ehigh"] - r["elow"]; r["share_pos"] = (x.y > 0).mean(); return r
    pt = stats(d); bs = pd.DataFrame([stats(d.iloc[rng.integers(0, len(d), len(d))]) for _ in range(2000)])
    for k in pt: out.append(dict(window=lab, stat=k, est=pt[k], lo=bs[k].quantile(.025), hi=bs[k].quantile(.975)))
    print(lab, len(d))
O = pd.DataFrame(out); O.to_csv(H + r"/results_vocab/site_logchange_boot.csv", index=False); print(O.round(3).to_string())
