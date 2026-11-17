"""Ex-ante edit share: only edits dated before the ChatGPT release.
ea_any  : answers posted 2021-01..2022-11, share with a body edit dated < 2022-12-01
ea30    : answers posted 2021-01..2022-10, share edited within 30 days of posting (all such edits precede 2022-12)
ea30_self / ea30_other : same, by self / other
Re-run: site screen (+placebo), terciles with bootstrap, code x edit cells, monthly panel (trend/no trend, placebo), RI,
joint with size/len/math, unedited-answer split. Output results_vocab/exante_*.csv"""
import glob, numpy as np, pandas as pd, statsmodels.formula.api as smf
H = str(__import__("pathlib").Path(__file__).resolve().parents[1]); R = H + r"/results_vocab"
X = pd.concat([pd.read_parquet(f) for f in glob.glob(H + r"/data_edit/edits_*.parquet")])
X["d"] = pd.to_datetime(X.date.str[:19]); CUT = pd.Timestamp("2022-12-01")
fs = pd.to_datetime(X.first_self.str[:19], errors="coerce"); fo = pd.to_datetime(X.first_other.str[:19], errors="coerce")
first = pd.concat([fs, fo], axis=1).min(axis=1)
X["ea_any"] = first.lt(CUT); X["e30"] = (first - X.d).dt.days.le(30)
X["e30_self"] = (fs - X.d).dt.days.le(30); X["e30_other"] = (fo - X.d).dt.days.le(30); X["ever"] = first.notna()
A = X[(X.date >= "2021-01") & (X.date < "2022-12")]; B = X[(X.date >= "2021-01") & (X.date < "2022-11")]
S = pd.DataFrame({"ea_any": A.groupby("site").ea_any.mean(), "ever": A.groupby("site").ever.mean(),
                  "ea30": B.groupby("site").e30.mean(), "ea30_self": B.groupby("site").e30_self.mean(), "ea30_other": B.groupby("site").e30_other.mean()})
E = pd.read_csv(R + r"/exposure_plus.csv").set_index("site"); S = S.join(E[["edited"]], how="inner")
print(S.describe().round(3)); print(S.corr(method="spearman").round(3))
# share of ever-edits (2021-22 answers) that happened after the cutoff
print("share of ever-edited 2021-22 answers whose first edit came after 2022-12-01:", round(1 - A.ea_any.sum() / A.ever.sum(), 3))
for c in ["ea_any", "ea30", "ea30_self", "ea30_other", "edited"]: S["r_" + c] = S[c].rank(pct=True) - .5
S.to_csv(R + r"/exante_edit.csv")
rows = []
# --- site screen
L = pd.read_csv(R + r"/site_logchange2.csv").set_index("site")[["y", "w", "rank", "r_size", "r_len"]].join(S, how="inner")
D = pd.read_csv(R + r"/ddd_panel.csv")
plA = ["2019Q1", "2019Q2", "2019Q3", "2019Q4", "2020Q1", "2020Q2", "2020Q3"]; plB = ["2021Q1", "2021Q2", "2021Q3", "2021Q4"]
g = D[D.q.isin(plA)].groupby("site").agg(mk=("mk", "mean"), ctl=("ctl", "mean"), n=("n", "sum")); h = D[D.q.isin(plB)].groupby("site").agg(mk=("mk", "mean"), ctl=("ctl", "mean"), n=("n", "sum"))
Pl = pd.DataFrame({"y": np.log(h.mk / g.mk) - np.log(h.ctl / g.ctl), "w": np.sqrt(h.n + g.n)}).replace([np.inf, -np.inf], np.nan).dropna().join(S, how="inner").join(E[["rank"]])
def fit(f, d, lab):
    m = smf.wls("y ~ " + f, d, weights=d.w).fit(cov_type="HC1")
    for k in f.split(" + "): rows.append(dict(model=lab, spec=f, term=k, b=m.params[k], p=m.pvalues[k], r2=m.rsquared, sites=len(d)))
for f in ["r_edited", "r_ea_any", "r_ea30", "rank + r_ea30", "r_ea30_self + r_ea30_other", "rank + r_ea30 + r_size + r_len"]:
    fit(f, L, "screen"); fit(f.replace(" + r_size + r_len", ""), Pl, "screen_placebo")
# terciles + bootstrap, code x edit cells
rng = np.random.default_rng(7)
for c in ["ea30", "ea_any"]:
    L["t"] = pd.qcut(L[c], 3, labels=[0, 1, 2]).astype(int)
    st = lambda x: [np.average(x.y[x.t == k], weights=x.w[x.t == k]) for k in range(3)]
    pt = st(L); bs = np.array([st(L.iloc[rng.integers(0, len(L), len(L))]) for _ in range(2000)])
    for k in range(3): rows.append(dict(model="tercile_" + c, spec="t%d" % k, term="mean", b=pt[k], p=np.nan, r2=bs[:, k].min() * 0 + np.quantile(bs[:, k], .025), sites=np.quantile(bs[:, k], .975)))
    hc = L["rank"] > 0; le = L[c] < L[c].median()
    for nm, m in {"hicode_loedit": hc & le, "hicode_hiedit": hc & ~le, "locode_loedit": ~hc & le, "locode_hiedit": ~hc & ~le}.items():
        rows.append(dict(model="cells_" + c, spec=nm, term="mean", b=np.average(L.y[m], weights=L.w[m]), p=np.nan, r2=np.nan, sites=int(m.sum())))
# --- monthly panel
P = pd.read_parquet(R + r"/panel_edit.parquet").merge(S[[c for c in S.columns if c.startswith("r_")]].drop(columns=["r_edited"]).reset_index(), on="site")
cl = lambda z: {"groups": z.site.astype("category").cat.codes}
for y in ["ext", "anyx"]:
    for tr in [False, True]:
        for pl in [False, True]:
            z = P.dropna(subset=[y]).copy()
            if pl: z = z[z.t < 0]; z["post"] = (z.t >= -12).astype(int)
            for terms in [["r_ea30"], ["r_ea_any"], ["rank", "r_ea30"], ["r_ea30_self", "r_ea30_other"], ["rank", "r_ea30", "r_size", "r_len", "r_math"]]:
                rhs = " + ".join(k + ":post" for k in terms)
                if terms == ["rank", "r_ea30"]: rhs += " + rank:r_ea30:post"
                m = smf.wls(f"{y} ~ C(site) + C(month) + {rhs}" + (" + C(site):t" if tr else ""), z, weights=np.minimum(z.n, 2000)).fit(cov_type="cluster", cov_kwds=cl(z))
                for k in [k for k in m.params.index if k.endswith(":post")]:
                    rows.append(dict(model=f"panel_{y} trend={tr} placebo={pl}", spec="+".join(terms), term=k[:-5], b=m.params[k], p=m.pvalues[k], r2=np.nan, sites=z.site.nunique()))
    print(y, flush=True)
# RI, ext, no trend, r_ea30 alone
z = P.reset_index(drop=True); w = np.sqrt(np.minimum(z.n, 2000).values)
Xd = np.hstack([pd.get_dummies(z.site).values, pd.get_dummies(z.month).values[:, 1:]]).astype(float) * w[:, None]; Q, _ = np.linalg.qr(Xd); res = lambda v: v - Q @ (Q.T @ v)
us = np.unique(z.site); pos = np.searchsorted(us, z.site.values); yt = res(z.ext.values * w)
rk = z.drop_duplicates("site").set_index("site")["r_ea30"].reindex(us).values; beta = lambda r: (lambda x: (x @ yt) / (x @ x))(res(r[pos] * z.post.values * w))
b0 = beta(rk); gg = np.random.default_rng(4); bsr = np.array([beta(gg.permutation(rk)) for _ in range(2000)])
rows.append(dict(model="RI ext notrend", spec="r_ea30", term="r_ea30", b=b0, p=(np.abs(bsr) >= abs(b0)).mean(), r2=np.nan, sites=len(us)))
# unedited answers split
U = pd.read_parquet(R + r"/panel_by_edited.parquet"); U = U[U.edited_flag != "share"] if "edited_flag" in U else U
U = U[U.edited_flag != "share"].copy(); U["t"] = U.month.map(lambda m: (int(m[:4]) - 2022) * 12 + int(m[5:7]) - 12); U = U[U.month.between("2019-01", "2024-03")]; U = U.merge(S[["r_ea30"]].reset_index(), on="site"); U["post"] = (U.t >= 0).astype(int)
for lab, zz in [("unedited", U[U.edited == False]), ("edited", U[U.edited == True])]:
    zz = zz.dropna(subset=["ext"])
    for pl in [False, True]:
        z2 = zz.copy()
        if pl: z2 = z2[z2.t < 0]; z2["post"] = (z2.t >= -12).astype(int)
        m = smf.wls("ext ~ C(site) + C(month) + r_ea30:post", z2, weights=np.minimum(z2.n, 2000)).fit(cov_type="cluster", cov_kwds=cl(z2))
        rows.append(dict(model=f"answers_{lab} placebo={pl}", spec="r_ea30", term="r_ea30", b=m.params["r_ea30:post"], p=m.pvalues["r_ea30:post"], r2=np.nan, sites=z2.site.nunique()))
O = pd.DataFrame(rows); O.to_csv(R + r"/exante_models.csv", index=False)
pd.set_option("display.width", 230); pd.set_option("display.max_rows", 400); print(O.round(4).to_string())
