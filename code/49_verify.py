"""Verify unsourced prose numbers: self/other tercile log changes; edit-share stability 2019 vs 2024 by tercile; other-edit share 2019 vs 2024."""
import glob, numpy as np, pandas as pd
H = str(__import__("pathlib").Path(__file__).resolve().parents[1]); R = H + r"/results_vocab"
S = pd.read_csv(R + r"/exante_edit.csv").set_index("site")
L = pd.read_csv(R + r"/site_logchange2.csv").set_index("site")[["y", "w"]].join(S, how="inner")
for c in ["ea30", "ea30_self", "ea30_other"]:
    L["t"] = pd.qcut(L[c], 3, labels=[0, 1, 2]).astype(int)
    print(c, [round(np.average(L.y[L.t == k], weights=L.w[L.t == k]), 3) for k in range(3)])
X = pd.concat([pd.read_parquet(f) for f in glob.glob(H + r"/data_edit/edits_*.parquet")])
X["d"] = pd.to_datetime(X.date.str[:19]); fs = pd.to_datetime(X.first_self.str[:19], errors="coerce"); fo = pd.to_datetime(X.first_other.str[:19], errors="coerce")
X["e30"] = (pd.concat([fs, fo], axis=1).min(axis=1) - X.d).dt.days.le(30); X["o30"] = (fo - X.d).dt.days.le(30); X["yr"] = X.date.str[:4]
X = X[X.d < pd.Timestamp("2024-03-01")]
g = X.groupby(["site", "yr"])[["e30", "o30"]].mean().unstack()
S["terc"] = pd.qcut(S.ea30, 3, labels=["low", "mid", "high"])
for t, h in S.groupby("terc", observed=True):
    print(t, "e30 2019", round(g["e30"]["2019"].reindex(h.index).median(), 3), "2024", round(g["e30"]["2024"].reindex(h.index).median(), 3), "2023", round(g["e30"]["2023"].reindex(h.index).median(), 3))
print("other 2019", round(g["o30"]["2019"].median(), 3), "2024", round(g["o30"]["2024"].median(), 3), "2023", round(g["o30"]["2023"].median(), 3))
print("median self/other", S.ea30_self.median(), S.ea30_other.median(), S.ea30.median(), S.ea30.quantile([.25, .75]).values)
