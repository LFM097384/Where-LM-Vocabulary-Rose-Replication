"""Referee checks for the edit gradient (existing data only). Output results_vocab/referee_*.csv
A composition by ea30 tercile (answers/month, authors, newcomer share) pre vs post
B tenure split panel: edit gradient for answers by established authors vs newcomers (first answer on site < 12 months earlier)
C screen with pre-level controls, higher answer thresholds, corr of edit rank with pre marker level and size
D leave-one-word-out and drop-three screens
E split-half screen of 15 traits (edited replaced by ea30), 500 splits
F event study with Jun-Nov / Dec-May half-years, reference Jun-Nov 2022, joint pre-test
G id-gap deletion proxy by site-month, edit gradient"""
import os, glob, numpy as np, pandas as pd, statsmodels.formula.api as smf
H = str(__import__("pathlib").Path(__file__).resolve().parents[1]); R = H + r"/results_vocab"; V = H + r"/vocab"; DA = H + r"/data_all"
S = pd.read_csv(R + r"/exante_edit.csv").set_index("site"); sites = list(S.index)
S["terc"] = pd.qcut(S.ea30, 3, labels=["low", "mid", "high"])
cl = lambda z: {"groups": z.site.astype("category").cat.codes}
out = []
def add(sec, **k): out.append(dict(section=sec, **k))

# ---------- answer-level frame
A = []
for s in sites:
    a = pd.read_parquet(DA + rf"/answers_{s}.parquet", columns=["id", "owner", "date"])
    m = pd.read_parquet(V + rf"/mk_{s}.parquet", columns=["id", "ntok", "ext_types"])
    a = a.merge(m, on="id", how="left"); a["site"] = s; A.append(a)
A = pd.concat(A, ignore_index=True); A["d"] = pd.to_datetime(A.date.str[:19])
A = A.sort_values("d"); fa = A[A.owner > 0].groupby(["site", "owner"]).d.min().rename("first").reset_index()
A = A.merge(fa, on=["site", "owner"], how="left")
A["newc"] = ((A.d - A["first"]).dt.days < 365) | (A.owner <= 0)
A["month"] = A.date.str[:7]; A = A[A.month.between("2019-01", "2024-03")].copy()
A["ext"] = 100 * A.ext_types / A.ntok.clip(lower=20)
A["t"] = A.month.map(lambda m: (int(m[:4]) - 2022) * 12 + int(m[5:7]) - 12)

# ---------- A composition
pre = A[A.month.between("2021-01", "2022-11")]; post = A[A.month.between("2023-03", "2024-03")]
for lab, z in [("pre", pre), ("post", post)]:
    nm = z.month.nunique()
    g = z.groupby("site").agg(ans=("id", "size"), auth=("owner", "nunique"), newc=("newc", "mean"))
    g["ans"] /= nm; g["auth"] /= nm; g = g.join(S.terc)
    for t, h in g.groupby("terc", observed=True):
        add("A_comp", period=lab, group=t, ans_med=h.ans.median(), ans_sum=h.ans.sum(), auth_sum=h.auth.sum(), newc_med=h.newc.median())
gp = pre.groupby("site").size() / pre.month.nunique(); gq = post.groupby("site").size() / post.month.nunique()
drop = np.log(gq / gp).rename("dlog_ans"); Cd = S.join(drop)
for t, h in Cd.groupby("terc", observed=True): add("A_drop", group=t, dlog_ans_med=h.dlog_ans.median())
nc = (post.groupby("site").newc.mean() - pre.groupby("site").newc.mean()).rename("d_newc"); Cd = Cd.join(nc)
for y in ["dlog_ans", "d_newc"]:
    m = smf.ols(f"{y} ~ r_ea30", Cd).fit(cov_type="HC1"); add("A_reg", outcome=y, b=m.params["r_ea30"], p=m.pvalues["r_ea30"])
print("A done", flush=True)

# ---------- B tenure split panel
def panel(z, y="ext", rhs="r_ea30:post", plac=False, trend=False):
    z = z.copy()
    if plac: z = z[z.t < 0]; z["post"] = (z.t >= -12).astype(int)
    m = smf.wls(f"{y} ~ C(site) + C(month) + {rhs}" + (" + C(site):t" if trend else ""), z, weights=np.minimum(z.n, 2000)).fit(cov_type="cluster", cov_kwds=cl(z))
    k = rhs.split(" + ")[0]; return m.params[k], m.pvalues[k]
for lab, z in [("established", A[~A.newc]), ("newcomer", A[A.newc]), ("all", A)]:
    P = z.dropna(subset=["ext"]).groupby(["site", "month", "t"]).agg(ext=("ext", "mean"), n=("id", "size")).reset_index()
    P = P.merge(S[["r_ea30"]].reset_index(), on="site"); P["post"] = (P.t >= 0).astype(int)
    for pl in [False, True]:
        for tr in [False, True]:
            b, p = panel(P, plac=pl, trend=tr); add("B_tenure", group=lab, placebo=pl, trend=tr, b=b, p=p, sites=P.site.nunique())
    print("B", lab, flush=True)
# composition-fixed reweighting: site-month marker rate with newcomer share held at site pre mean
P2 = A.dropna(subset=["ext"]).groupby(["site", "month", "t", "newc"]).agg(ext=("ext", "mean"), n=("id", "size")).reset_index()
w0 = A[A.month.between("2021-01", "2022-11")].groupby("site").newc.mean().rename("w0")
Q = P2.pivot_table(index=["site", "month", "t"], columns="newc", values="ext").rename(columns={False: "e_old", True: "e_new"}).reset_index()
Q = Q.merge(P2.groupby(["site", "month"]).n.sum().reset_index(), on=["site", "month"]).merge(w0.reset_index(), on="site").dropna()
Q["ext"] = Q.w0 * Q.e_new + (1 - Q.w0) * Q.e_old; Q = Q.merge(S[["r_ea30"]].reset_index(), on="site"); Q["post"] = (Q.t >= 0).astype(int)
for pl in [False, True]:
    b, p = panel(Q, plac=pl); add("B_reweight", placebo=pl, b=b, p=p, sites=Q.site.nunique())
print("B done", flush=True)

# ---------- C/D/E site screen machinery (rebuild marker and control scores as in 26_ddd.py)
src = open(H + r"/code/07_vocab.py", encoding="utf8").read(); i = src.index("EXT = set("); j = src.index(".split())", i) + 9; EXT = sorted(eval(src[i + 6:j]))
QS = [f"{y}Q{k}" for y in range(2019, 2025) for k in range(1, 5) if f"{y}Q{k}" <= "2024Q1"]; FIT = [q for q in QS if "2021Q1" <= q <= "2022Q3"]
vs, ns = [], []
for s in sites:
    f = V + rf"/vocab_{s}.parquet"
    if not os.path.exists(f): continue
    v = pd.read_parquet(f); v = v[v.q.isin(QS)]; v["site"] = s; vs.append(v)
    n = pd.read_parquet(V + rf"/nq_{s}.parquet"); n = n[n.q.isin(QS)]; n["site"] = s; ns.append(n)
N = pd.concat(ns); V_ = pd.concat(vs)
prev = V_[V_.q.isin(FIT)]; nsites = prev.groupby("word").site.nunique(); tot = prev.groupby("word").df.sum() / N[N.q.isin(FIT)].n.sum()
pool = tot[(nsites >= 50) & tot.index.str.fullmatch(r"[a-z]{4,}") & ~tot.index.isin(EXT)]; mk = tot.reindex([w for w in EXT if w in tot.index]).dropna()
bins = np.quantile(pool.values, np.linspace(0, 1, 21)); pb = np.clip(np.digitize(pool.values, bins[1:-1]), 0, 19); mb = np.clip(np.digitize(mk.values, bins[1:-1]), 0, 19)
W = V_[V_.word.isin(set(pool.index) | set(EXT))].pivot_table(index=["site", "q"], columns="word", values="df", aggfunc="sum", fill_value=0)
base = N.set_index(["site", "q"]).n; base = base[base >= 20]; W = W.reindex(base.index, fill_value=0); Sh = 100 * W.div(base, axis=0)
ci = {w: k for k, w in enumerate(Sh.columns)}; Ar = Sh.values; sc = lambda l: Ar[:, [ci[w] for w in l if w in ci]].mean(axis=1)
rng = np.random.default_rng(5); pw = pool.index.values
ctl = np.mean([sc([rng.choice(pw[pb == b_]) for b_ in mb]) for _ in range(200)], axis=0)
Pq = base.rename("n").reset_index(); Pq["ctl"] = ctl
preq = ["2021Q1", "2021Q2", "2021Q3", "2021Q4", "2022Q1", "2022Q2", "2022Q3"]; postq = ["2023Q2", "2023Q3", "2023Q4", "2024Q1"]
def logchg(mkv, minn=20):
    z = Pq.copy(); z["mk"] = mkv
    g = z[z.q.isin(preq)].groupby("site").agg(mk=("mk", "mean"), ctl=("ctl", "mean"), n=("n", "sum"))
    h = z[z.q.isin(postq)].groupby("site").agg(mk=("mk", "mean"), ctl=("ctl", "mean"), n=("n", "sum"))
    d = pd.DataFrame({"y": np.log(h.mk / g.mk) - np.log(h.ctl / g.ctl), "w": np.sqrt(h.n + g.n), "npre": g.n, "npost": h.n,
                      "lpre_mk": np.log(g.mk), "lpre_rel": np.log(g.mk / g.ctl)}).replace([np.inf, -np.inf], np.nan).dropna()
    return d[(d.npre >= minn) & (d.npost >= minn)]
L0 = pd.read_csv(R + r"/site_logchange2.csv").set_index("site")
mfull = sc([w for w in EXT if w in ci]); d0 = logchg(mfull).join(S[["r_ea30"]]).join(L0[["rank", "r_size", "r_len"]])
print("rebuilt screen sites", len(d0), "corr with saved y", np.corrcoef(d0.y, L0.y.reindex(d0.index))[0, 1], flush=True)
def scr(d, f, lab, **k):
    m = smf.wls("y ~ " + f, d, weights=d.w).fit(cov_type="HC1"); add(lab, spec=f, b=m.params["r_ea30"], p=m.pvalues["r_ea30"], r2=m.rsquared, sites=len(d), **k)
scr(d0, "r_ea30", "C_base")
scr(d0, "r_ea30 + lpre_mk", "C_prelevel"); scr(d0, "r_ea30 + lpre_rel", "C_prelevel"); scr(d0, "r_ea30 + lpre_mk + lpre_rel + rank + r_size + r_len", "C_prelevel")
for mn in [100, 200, 500]: scr(logchg(mfull, mn).join(S[["r_ea30"]]), "r_ea30", "C_threshold", minn=mn)
z = d0.join(S[["ea30"]]); add("C_corr", what="spearman r_ea30 vs lpre_mk", b=z.r_ea30.corr(z.lpre_mk, method="spearman"))
add("C_corr", what="spearman r_ea30 vs lpre_rel", b=z.r_ea30.corr(z.lpre_rel, method="spearman")); add("C_corr", what="spearman r_ea30 vs r_size", b=z.r_ea30.corr(z.r_size, method="spearman"))
# unweighted
m = smf.ols("y ~ r_ea30", d0).fit(cov_type="HC1"); add("C_unweighted", spec="r_ea30", b=m.params["r_ea30"], p=m.pvalues["r_ea30"], r2=m.rsquared, sites=len(d0))
# D leave-one-word-out
top = pd.read_csv(R + r"/word_edit_gradient_exante.csv").word.tolist()
for w in top + [("ensure", "essential", "crucial"), ("ensure", "essential", "furthermore"), ("ensure", "essential", "furthermore", "crucial", "additionally", "moreover")]:
    ex = set([w] if isinstance(w, str) else w); d = logchg(sc([x for x in EXT if x in ci and x not in ex])).join(S[["r_ea30"]])
    scr(d, "r_ea30", "D_drop", dropped="+".join(sorted(ex)))
print("D done", flush=True)
# E split-half of the 15-trait screen with ea30 replacing the dump flag
E_ = pd.read_csv(H + r"/results_all/exposure.csv").set_index("site")
pn = pd.read_csv(H + r"/results_all/panel.csv"); pp = pn[pn.month.between("2021-01", "2022-11")]
T = pp.groupby("site").apply(lambda g: pd.Series({c: np.average(g[c].fillna(g[c].mean()), weights=g.n.fillna(0) + 1e-9) for c in ["list", "head", "lw_sd", "lqw_m", "heres", "summary", "dv"]}))
T["lnq"] = np.log1p(pp.groupby("site").nq.mean()); T["apq"] = pp.groupby("site").n.sum() / pp.groupby("site").nq.sum()
T = T.join(E_[["code_q", "code_a", "math_a", "n_pre", "lw_pre"]]).join(S[["ea30"]]); Tr = T.rank(pct=True) - .5
DS = d0[["y", "w"]].join(Tr, how="inner"); tr_ = list(T.columns); print("traits", len(tr_), tr_, flush=True)
g = np.random.default_rng(11); top1 = []; rep = []
for it in range(500):
    idx = g.permutation(len(DS)); a_, b_ = DS.iloc[idx[: len(DS) // 2]], DS.iloc[idx[len(DS) // 2:]]
    r2 = {c: smf.wls(f"y ~ {c}", a_, weights=a_.w).fit().rsquared for c in tr_}; best = max(r2, key=r2.get); top1.append(best)
    mb_ = smf.wls(f"y ~ {best}", b_, weights=b_.w).fit(cov_type="HC1"); ma_ = smf.wls(f"y ~ {best}", a_, weights=a_.w).fit()
    rep.append(dict(best=best, same_sign=np.sign(mb_.params[best]) == np.sign(ma_.params[best]), sig=mb_.pvalues[best] < .05 and np.sign(mb_.params[best]) == np.sign(ma_.params[best])))
rep = pd.DataFrame(rep)
for c, h in rep.groupby("best"): add("E_split", trait=c, share_top=len(h) / 500, rep_sign=h.same_sign.mean(), rep_sig=h.sig.mean())
add("E_split", trait="ALL", share_top=1.0, rep_sign=rep.same_sign.mean(), rep_sig=rep.sig.mean())
full = {c: smf.wls(f"y ~ {c}", DS, weights=DS.w).fit(cov_type="HC1") for c in tr_}
for c, m in full.items(): add("E_full", trait=c, b=m.params[c], p=m.pvalues[c], r2=m.rsquared, sites=len(DS))
print("E done", flush=True)

# ---------- F event study with Dec-May / Jun-Nov half-years, reference Jun-Nov 2022
P = pd.read_parquet(R + r"/panel_edit.parquet").merge(S[["r_ea30"]].reset_index(), on="site")
P["hb"] = np.floor(P.t / 6).astype(int)  # -1 = 2022-06..2022-11
terms = []
for h in sorted(P.hb.unique()):
    if h == -1: continue
    c = f"e_{'m' if h < 0 else 'p'}{abs(h)}"; P[c] = (P.hb == h) * P.r_ea30; terms.append((h, c))
m = smf.wls("ext ~ C(site) + C(month) + " + " + ".join(c for _, c in terms), P, weights=np.minimum(P.n, 2000)).fit(cov_type="cluster", cov_kwds=cl(P))
for h, c in terms: add("F_event", half=h, b=m.params[c], se=m.bse[c], p=m.pvalues[c])
prec = [c for h, c in terms if h < -1]; ft = m.f_test(" , ".join(f"{c} = 0" for c in prec)); add("F_pretest", b=float(np.squeeze(ft.fvalue)), p=float(ft.pvalue), what=f"{len(prec)} pre coefs")
print("F done", flush=True)

# ---------- G id-gap deletion proxy
G = []
for s in sites:
    q = pd.read_parquet(DA + rf"/questions_{s}.parquet", columns=["id", "qdate"]).rename(columns={"qdate": "date"})
    a = pd.read_parquet(DA + rf"/answers_{s}.parquet", columns=["id", "date"]); z = pd.concat([q, a]).sort_values("id")
    z = z[z.date >= "2018-12"]; z["month"] = z.date.str[:7]
    # ids are allocated in time order; ids in a month span from first to next month's first id
    f = z.groupby("month").id.min(); span = f.shift(-1) - f; obs = z.groupby("month").size()
    g_ = pd.DataFrame({"span": span, "obs": obs}).dropna(); g_ = g_[g_.span > 0]; g_["gap"] = 1 - g_.obs / g_.span; g_["site"] = s; G.append(g_.reset_index())
G = pd.concat(G); G = G[G.month.between("2019-01", "2024-02")]; G["t"] = G.month.map(lambda m: (int(m[:4]) - 2022) * 12 + int(m[5:7]) - 12)
G = G.merge(S[["r_ea30"]].reset_index(), on="site"); G["post"] = (G.t >= 0).astype(int); G["n"] = G.span; G = G.rename(columns={"gap": "gapr"})
for lab, z in [("2022-12..2023-08", G[(G.t < 9)]), ("all", G)]:
    for pl in [False, True]:
        b, p = panel(z, y="gapr", plac=pl); add("G_idgap", window=lab, placebo=pl, b=b, p=p, sites=z.site.nunique())
gm = G.groupby(G.t >= 0).gapr.mean(); add("G_mean", what="mean gap pre/post", b=gm.get(False), p=gm.get(True))
for t_, h in G.merge(S.terc.reset_index(), on="site").groupby("terc", observed=True):
    add("G_terc", group=t_, b=h[h.t < 0].gapr.mean(), p=h[(h.t >= 0) & (h.t < 9)].gapr.mean())
O = pd.DataFrame(out); O.to_csv(R + r"/referee_checks.csv", index=False)
pd.set_option("display.width", 250); pd.set_option("display.max_rows", 500); pd.set_option("display.max_columns", 30); print(O.round(4).to_string())

