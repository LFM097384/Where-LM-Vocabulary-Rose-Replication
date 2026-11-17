"""Extract answers 2019-01..2023-09 from the 2023-09-12 snapshot (archive.org item stackexchange_20230912).
Per answer: id, owner, date, wc, nli, nhd, ntok, ext_tok, ext_types. Output snap/snap_<site>.parquet. Usage: <k> <n>"""
import os, sys, re, json, time, shutil, traceback, importlib.util
import pandas as pd, requests, py7zr
from lxml import etree
H = str(__import__("pathlib").Path(__file__).resolve().parents[1])
sp = importlib.util.spec_from_file_location("a4", H + r"/code/04_allsites.py"); a4 = importlib.util.module_from_spec(sp); sp.loader.exec_module(a4)
src = open(H + r"/code/07_vocab.py", encoding="utf8").read(); i = src.index("EXT = set("); j = src.index(".split())", i) + 9; EXT = eval(src[i + 6:j])
TOK = re.compile(r"[a-z]+(?:'[a-z]+)?")
ITEM = "stackexchange_20230912"; TMP = H + r"/raw_snap"; OUT = H + r"/snap"; os.makedirs(TMP, exist_ok=True); os.makedirs(OUT, exist_ok=True)
def sites():
    m = json.load(open(H + rf"/meta/meta_{ITEM}.json")); out = []
    for f in m["files"]:
        n = f["name"]
        if n.endswith(".7z") and "stackoverflow" not in n.lower() and ".meta." not in n and not n.startswith("meta."): out.append((n[:-3], int(f["size"])))
    return sorted(out, key=lambda x: -x[1])
def download(name, total):
    url = f"https://archive.org/download/{ITEM}/{name}.7z"; out = os.path.join(TMP, name + ".7z")
    for attempt in range(40):
        have = os.path.getsize(out) if os.path.exists(out) else 0
        if have >= total: return out
        try:
            with requests.get(url, headers={"Range": f"bytes={have}-"} if have else {}, stream=True, timeout=120) as r:
                r.raise_for_status(); mode = "ab" if (have and r.status_code == 206) else "wb"
                with open(out, mode) as f:
                    for c in r.iter_content(1 << 20): f.write(c)
        except Exception as e: print("retry", name, attempt, repr(e)[:150], flush=True); time.sleep(20)
    raise RuntimeError("download failed")
def process(name, size):
    short = name.replace(".stackexchange.com", "").replace(".com", "").replace(".net", "")
    of = os.path.join(OUT, f"snap_{short}.parquet")
    if os.path.exists(of): return
    p = download(name, size); d = os.path.join(TMP, name + "_x")
    with py7zr.SevenZipFile(p, "r") as z:
        tg = [n for n in z.getnames() if n.lower().endswith("posts.xml")]
    with py7zr.SevenZipFile(p, "r") as z: z.extract(path=d, targets=tg)
    xml = os.path.join(d, tg[0]); rows = []
    for _, el in etree.iterparse(xml, tag="row", huge_tree=True):
        if el.get("PostTypeId") == "2":
            dt = el.get("CreationDate", "")
            if dt >= "2019-01-01":
                body = el.get("Body", ""); txt = a4.clean(body); toks = TOK.findall(txt.lower()); s = set(toks)
                rows.append((int(el.get("Id")), int(el.get("OwnerUserId", -1)), dt, len(txt.split()), len(a4.LI.findall(body)),
                             len(a4.HD.findall(body)), len(toks), sum(t in EXT for t in toks), len(s & EXT)))
        el.clear()
        while el.getprevious() is not None: del el.getparent()[0]
    df = pd.DataFrame(rows, columns=["id", "owner", "date", "wc", "nli", "nhd", "ntok", "ext_tok", "ext_types"]); df["site"] = short
    df.to_parquet(of + ".tmp", index=False); os.replace(of + ".tmp", of)
    shutil.rmtree(d, ignore_errors=True); os.remove(p); print(time.strftime("%H:%M"), "done", short, len(df), flush=True)
if __name__ == "__main__":
    k, n = int(sys.argv[1]), int(sys.argv[2]); S = sites()[k::n]; print("worker", k, len(S), flush=True)
    for name, size in S:
        try: process(name, size)
        except Exception: print("FAIL", name, flush=True); traceback.print_exc()
    print("WORKER DONE", k, flush=True)
