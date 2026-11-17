"""All-site extension. For every main SE site in the 2024-04 dump (excluding stackoverflow.com and meta sites):
download the 7z, extract Posts.xml, extract answer/question features for 2017-07..2024-03, write parquet, delete 7z+xml.
Usage: python 04_allsites.py <worker_index> <n_workers>"""
import os, sys, re, html, json, time, shutil, traceback
import numpy as np, pandas as pd, requests, py7zr
from lxml import etree
BASE = str(__import__("pathlib").Path(__file__).resolve().parents[1])
TMP = os.path.join(BASE, "raw_all"); OUT = os.path.join(BASE, "data_all")
os.makedirs(TMP, exist_ok=True); os.makedirs(OUT, exist_ok=True)
META = str(__import__("pathlib").Path(__file__).resolve().parents[1]) + "/meta/meta_stackexchange_20240402.json"
OLD = str(__import__("pathlib").Path(__file__).resolve().parents[1]) + "/raw_all"
PRE = re.compile(r"<pre[^>]*>.*?</pre>", re.S | re.I); CODE = re.compile(r"<code[^>]*>.*?</code>", re.S | re.I)
TAG = re.compile(r"<[^>]+>"); URL = re.compile(r"https?://\S+"); WS = re.compile(r"\s+")
LI = re.compile(r"<li[\s>]", re.I); HD = re.compile(r"<h[1-6][\s>]", re.I); P = re.compile(r"<p[\s>]", re.I)
B = re.compile(r"<(strong|b)[\s>]", re.I); OL = re.compile(r"<ol[\s>]", re.I)
MATH = re.compile(r"\$[^$]{1,200}\$")
PHR = {"heres": re.compile(r"\bhere(?:'|’)?s (?:a|an|how|what|the|some)\b", re.I),
       "summary": re.compile(r"\b(?:in summary|to summarize|in conclusion|overall,)", re.I),
       "hope": re.compile(r"\bhope (?:this|that) helps\b", re.I),
       "delve": re.compile(r"\b(?:delve|delves|delving|crucial|ensure|additionally|furthermore|moreover)\b", re.I)}
def clean(b):
    t = PRE.sub(" ", b); t = CODE.sub(" ", t); t = TAG.sub(" ", t); t = html.unescape(t); t = URL.sub(" ", t)
    return WS.sub(" ", t).strip()

def sites():
    m = json.load(open(META, encoding="utf8")); fs = m["files"] if "files" in m else m
    out = []
    for f in fs:
        n = f["name"]
        if not n.endswith(".7z") or "stackoverflow.com" in n or ".meta." in n or n.startswith("meta."): continue
        out.append((n[:-3], int(f["size"])))
    return sorted(out, key=lambda x: -x[1])

def download(name, total):
    url = f"https://archive.org/download/stackexchange/{name}.7z"; out = os.path.join(TMP, name + ".7z")
    for attempt in range(30):
        have = os.path.getsize(out) if os.path.exists(out) else 0
        if have >= total: return out
        try:
            hdr = {"Range": f"bytes={have}-"} if have else {}
            with requests.get(url, headers=hdr, stream=True, timeout=120, allow_redirects=True) as r:
                r.raise_for_status()
                mode = "ab" if (have and r.status_code == 206) else "wb"
                with open(out, mode) as f:
                    for chunk in r.iter_content(1 << 20): f.write(chunk)
        except Exception as e:
            print("retry", name, attempt, repr(e)[:200], flush=True); time.sleep(15)
    raise RuntimeError("download failed " + name)

def parse(xml, short):
    A, Q = [], []
    for _, el in etree.iterparse(xml, tag="row", events=("end",), huge_tree=True):
        t = el.get("PostTypeId"); d = el.get("CreationDate", "")
        if t in ("1", "2") and d >= "2017-07-01":
            body = el.get("Body", ""); txt = clean(body); wc = len(txt.split())
            if t == "2":
                r = dict(id=int(el.get("Id")), parent=int(el.get("ParentId", -1)), owner=int(el.get("OwnerUserId", -1)),
                         date=d, edited=el.get("LastEditDate") is not None, score=int(el.get("Score", 0)), wc=wc,
                         nchar=len(txt), npre=len(PRE.findall(body)), ncode=len(CODE.findall(body)), nmath=len(MATH.findall(txt)),
                         nli=len(LI.findall(body)), nol=len(OL.findall(body)), nhd=len(HD.findall(body)),
                         np=len(P.findall(body)), nb=len(B.findall(body)))
                for k, rx in PHR.items(): r[k] = len(rx.findall(txt))
                A.append(r)
            else:
                Q.append(dict(id=int(el.get("Id")), qdate=d, qwc=wc, qtags=el.get("Tags", ""), qviews=int(el.get("ViewCount", 0) or 0),
                              qanswers=int(el.get("AnswerCount", 0) or 0), qowner=int(el.get("OwnerUserId", -1)),
                              qnpre=len(PRE.findall(body))))
        el.clear()
        while el.getprevious() is not None: del el.getparent()[0]
    a = pd.DataFrame(A); q = pd.DataFrame(Q); a["site"] = short; q["site"] = short
    return a, q

def process(name, size):
    short = name.replace(".stackexchange.com", "").replace(".com", "").replace(".net", "")
    of = os.path.join(OUT, f"answers_{short}.parquet")
    if os.path.exists(of): return
    old = os.path.join(OLD, name + ".Posts.xml")
    if os.path.exists(old):
        xml = old; cleanup = False
    else:
        p = download(name, size); d = os.path.join(TMP, name + "_x")
        with py7zr.SevenZipFile(p, "r") as z: z.extract(path=d, targets=["Posts.xml"])
        xml = os.path.join(d, "Posts.xml"); cleanup = True
    a, q = parse(xml, short)
    q.to_parquet(os.path.join(OUT, f"questions_{short}.parquet"), index=False)
    a.to_parquet(of + ".tmp", index=False); os.replace(of + ".tmp", of)
    if cleanup:
        shutil.rmtree(os.path.join(TMP, name + "_x"), ignore_errors=True); os.remove(os.path.join(TMP, name + ".7z"))
    print(time.strftime("%H:%M"), "done", short, len(a), len(q), flush=True)

if __name__ == "__main__":
    k, n = int(sys.argv[1]), int(sys.argv[2])
    S = sites()[k::n]
    print("worker", k, "sites", len(S), flush=True)
    for name, size in S:
        try: process(name, size)
        except Exception: print("FAIL", name, flush=True); traceback.print_exc()
    print("WORKER DONE", k, flush=True)
