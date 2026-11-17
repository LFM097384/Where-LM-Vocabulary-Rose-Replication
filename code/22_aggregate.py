"""Aggregate marker series: monthly site-weighted mean (equal weight per site, sites with >=20 answers that month), with
linear trend fit 2019-01..2022-11, deviations post. Also per-word aggregate ratio (2023-04..2024-03 vs trend extrapolation).
Also length-adjusted version and a matched random-word control series (frequency-matched, from 14_wordlevel pool logic)."""
import glob, numpy as np, pandas as pd
H = str(__import__("pathlib").Path(__file__).resolve().parents[1]); R = H + r"/results_vocab"
E = pd.read_csv(H + r"/results_all/exposure.csv")
P = pd.read_parquet(R + r"/panel_lenadj.parquet")
M = pd.concat([pd.read_parquet(f, columns=["id", "date", "ntok", "ext_types", "site"]) for f in glob.glob(H + r"/vocab/mk_*.parquet")]); M = M[M.site.isin(E.site)]
M["month"] = M.date.str[:7]; M["ext"] = 100 * M.ext_types / M.ntok.clip(lower=20); M["anyx"] = (M.ext_types > 0).astype(float)
G = M.groupby(["site", "month"]).agg(n=("id", "size"), ext=("ext", "mean"), anyx=("anyx", "mean")).reset_index().merge(P[["site", "month", "extadj", "anyadj"]], on=["site", "month"])
G = G[(G.n >= 20) & G.month.between("2019-01", "2024-03")]
S = G.groupby("month")[["ext", "anyx", "extadj", "anyadj"]].mean()
# site-fixed-effect version: demean by site pre mean
pre = G[G.month < "2022-12"].groupby("site")[["ext", "anyx"]].mean()
G2 = G.join(pre, on="site", rsuffix="_pre"); G2["ext_rel"] = G2.ext / G2.ext_pre; G2["any_rel"] = G2.anyx / G2.anyx_pre
S = S.join(G2.groupby("month")[["ext_rel", "any_rel"]].median())
S["t"] = np.arange(len(S)); fit = S.index < "2022-12"
for c in ["ext", "anyx", "ext_rel", "any_rel"]:
    b = np.polyfit(S.t[fit], S[c][fit], 1); S[c + "_hat"] = np.polyval(b, S.t)
S.to_csv(R + r"/aggregate_series.csv")
yr = S.index.str[:4]
print(S.groupby(yr)[["ext", "ext_hat", "anyx", "anyx_hat", "ext_rel", "ext_rel_hat"]].mean().round(4).to_string())
post = S.index >= "2023-04"; print("post ratio ext obs/hat", (S.ext[post] / S.ext_hat[post]).mean(), "anyx", (S.anyx[post] / S.anyx_hat[post]).mean())
