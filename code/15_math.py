"""Falsification dose (math share) and horse race with code rank, for marker outcomes and list share."""
import glob, numpy as np, pandas as pd, statsmodels.formula.api as smf
H = str(__import__("pathlib").Path(__file__).resolve().parents[1])
E = pd.read_csv(H + r"/results_all/exposure.csv"); E["rank"] = E.code_q.rank(pct=True) - .5; E["mrank"] = E.math_a.rank(pct=True) - .5
M = pd.concat([pd.read_parquet(f, columns=["id", "date", "ntok", "ext_types", "site"]) for f in glob.glob(H + r"/vocab/mk_*.parquet")]); M = M[M.site.isin(E.site)]
M["month"] = M.date.str[:7]; M["ext"] = 100 * M.ext_types / M.ntok.clip(lower=20); M["anyx"] = (M.ext_types > 0).astype(float)
P = M.groupby(["site", "month"]).agg(n=("id", "size"), ext=("ext", "mean"), anyx=("anyx", "mean")).reset_index()
L = pd.read_csv(H + r"/results_all/panel.csv")[["site", "month", "list", "head", "lw"]]; P = P.merge(L, on=["site", "month"], how="left")
P = P[P.month.between("2019-01", "2024-03")]; P["t"] = P.month.map(lambda m: (int(m[:4]) - 2022) * 12 + int(m[5:7]) - 12); P = P.merge(E[["site", "rank", "mrank"]], on="site")
P["post"] = (P.t >= 0).astype(int)
out = []
for y in ["ext", "anyx", "list", "head"]:
    z = P.dropna(subset=[y])
    for f, lab in [("mrank:post", "math only"), ("rank:post + mrank:post", "horse race")]:
        m = smf.wls(f"{y} ~ C(site) + C(month) + {f} + C(site):t", z, weights=np.minimum(z.n, 2000)).fit(cov_type="cluster", cov_kwds={"groups": z.site.astype("category").cat.codes})
        for k in [k for k in ["rank:post", "mrank:post"] if k in m.params]: out.append(dict(outcome=y, spec=lab, term=k, b=m.params[k], p=m.pvalues[k]))
O = pd.DataFrame(out); O.to_csv(H + r"/results_vocab/math_falsification.csv", index=False); print(O.round(4).to_string())
print("corr code/math rank", E[["rank", "mrank"]].corr().iloc[0, 1])
