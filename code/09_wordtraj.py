import glob, os, numpy as np, pandas as pd
B = str(__import__("pathlib").Path(__file__).resolve().parents[1]); V = B + r"/vocab"
exec(open(B + r"/code/07_vocab.py", encoding="utf8").read().split("def quarter")[0].split("EXT = ")[1].join(["EXT = ", ""]) if False else "")
src = open(B + r"/code/07_vocab.py", encoding="utf8").read(); i = src.index("EXT = set("); j = src.index(".split())", i) + 9; EXT = eval(src[i + 6:j])
E = pd.read_csv(B + r"/results_all/exposure.csv"); E["hi"] = E.code_q.rank(pct=True) > .5
rows = []
for f in glob.glob(V + r"/vocab_*.parquet"):
    s = os.path.basename(f)[6:-8]
    if s not in set(E.site): continue
    v = pd.read_parquet(f); n = pd.read_parquet(V + rf"/nq_{s}.parquet").set_index("q").n
    v = v[v.word.isin(EXT)].copy(); v["sh"] = v.df / v.q.map(n)
    pre = v[v.q.between("2021Q1", "2022Q3")].groupby("word").sh.mean(); post = v[v.q.between("2023Q2", "2024Q1")].groupby("word").sh.mean()
    p0 = v[v.q.between("2019Q1", "2020Q3")].groupby("word").sh.mean()
    d = pd.DataFrame({"pre0": p0, "pre": pre, "post": post}).fillna(0); d["site"] = s; d["hi"] = bool(E.set_index("site").hi[s]); rows.append(d.reset_index())
D = pd.concat(rows); g = D.groupby(["word", "hi"])[["pre0", "pre", "post"]].mean().unstack()
out = pd.DataFrame({"r_post_lo": g[("post", False)] / g[("pre", False)], "r_post_hi": g[("post", True)] / g[("pre", True)],
                    "r_placebo_lo": g[("pre", False)] / g[("pre0", False)], "r_placebo_hi": g[("pre", True)] / g[("pre0", True)],
                    "pre_hi_per1000": 1000 * g[("pre", True)]}).sort_values("r_post_hi", ascending=False)
out.to_csv(B + r"/results_vocab/external_word_ratios.csv"); pd.set_option("display.width", 200); print(out.round(2).to_string())
