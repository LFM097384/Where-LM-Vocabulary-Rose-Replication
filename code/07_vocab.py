"""Excess-vocabulary extraction for all SE sites.
Per site: (1) vocab_<s>.parquet: quarter, word, df (answers containing the word), for words with site total df>=10;
          (2) nq_<s>.parquet: answers per quarter;
          (3) mk_<s>.parquet: per answer id, owner, date, n tokens, count of external LLM-style marker words (types and tokens).
Answers 2019-01..2024-03, text cleaned as in 04_allsites.py. Usage: python 07_vocab.py <k> <n>"""
import os, sys, re, time, shutil, traceback, collections
import numpy as np, pandas as pd, py7zr
from lxml import etree
HERE = str(__import__("pathlib").Path(__file__).resolve().parents[1])
import importlib.util
spec = importlib.util.spec_from_file_location("a4", os.path.join(HERE, "code", "04_allsites.py")); a4 = importlib.util.module_from_spec(spec); spec.loader.exec_module(a4)
B = str(__import__("pathlib").Path(__file__).resolve().parents[1]); OUT = os.path.join(B, "vocab"); os.makedirs(OUT, exist_ok=True)
TOK = re.compile(r"[a-z]+(?:'[a-z]+)?")
# LLM-favoured style words, adapted from the excess-vocabulary lists of Kobak et al. and Liang et al. (fixed before analysis)
EXT = set("""delve delves delved delving showcase showcases showcasing underscore underscores underscoring crucial pivotal intricate
intricacies meticulous meticulously comprehensive notably realm tapestry multifaceted nuanced commendable invaluable seamless seamlessly
leverage leveraging streamline streamlining robust foster fostering enhance enhancing bolster garner garnered paramount unwavering
embark navigate navigating landscape testament elevate profound versatile noteworthy additionally furthermore moreover ensure ensuring
essential vital""".split())
def quarter(d): return d[:4] + "Q" + str((int(d[5:7]) - 1) // 3 + 1)
def process(name, size):
    short = name.replace(".stackexchange.com", "").replace(".com", "").replace(".net", "")
    of = os.path.join(OUT, f"mk_{short}.parquet")
    if os.path.exists(of): return
    old = os.path.join(a4.OLD, name + ".Posts.xml")
    if os.path.exists(old): xml, cleanup = old, False
    else:
        p = a4.download(name, size); d = os.path.join(a4.TMP, name + "_v")
        with py7zr.SevenZipFile(p, "r") as z: z.extract(path=d, targets=["Posts.xml"])
        xml, cleanup = os.path.join(d, "Posts.xml"), True
    DF = collections.defaultdict(collections.Counter); NQ = collections.Counter(); MK = []
    for _, el in etree.iterparse(xml, tag="row", events=("end",), huge_tree=True):
        if el.get("PostTypeId") == "2":
            dt = el.get("CreationDate", "")
            if "2019-01-01" <= dt < "2024-04-01":
                toks = TOK.findall(a4.clean(el.get("Body", "")).lower())
                if toks:
                    q = quarter(dt); s = set(toks); NQ[q] += 1; DF[q].update(s)
                    MK.append((int(el.get("Id")), int(el.get("OwnerUserId", -1)), dt, len(toks),
                               sum(1 for t in toks if t in EXT), len(s & EXT)))
        el.clear()
        while el.getprevious() is not None: del el.getparent()[0]
    tot = collections.Counter()
    for c in DF.values(): tot.update(c)
    keep = {w for w, c in tot.items() if c >= 10}
    V = pd.DataFrame([(q, w, c) for q, cc in DF.items() for w, c in cc.items() if w in keep], columns=["q", "word", "df"])
    V["df"] = V.df.astype("int32"); V.to_parquet(os.path.join(OUT, f"vocab_{short}.parquet"), index=False)
    pd.DataFrame(sorted(NQ.items()), columns=["q", "n"]).to_parquet(os.path.join(OUT, f"nq_{short}.parquet"), index=False)
    m = pd.DataFrame(MK, columns=["id", "owner", "date", "ntok", "ext_tok", "ext_types"]); m["site"] = short
    m.to_parquet(of + ".tmp", index=False); os.replace(of + ".tmp", of)
    if cleanup:
        shutil.rmtree(os.path.join(a4.TMP, name + "_v"), ignore_errors=True); os.remove(os.path.join(a4.TMP, name + ".7z"))
    print(time.strftime("%H:%M"), "done", short, len(m), len(V), flush=True)
if __name__ == "__main__":
    k, n = int(sys.argv[1]), int(sys.argv[2]); S = a4.sites()[k::n]; print("worker", k, len(S), flush=True)
    for name, size in S:
        try: process(name, size)
        except Exception: print("FAIL", name, flush=True); traceback.print_exc()
    print("WORKER DONE", k, flush=True)
