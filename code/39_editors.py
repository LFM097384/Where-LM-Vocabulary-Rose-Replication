"""Self vs other edits. For every panel site: download the 2024-04 7z, extract PostHistory.xml, keep body edits
(PostHistoryTypeId 5 = edit body, 8 = rollback body) on answers present in data_all/answers_<site>.parquet.
Per answer: n_self (editor == owner), n_other (editor != owner, incl. community user -1), first_other (date of first other edit),
first_self. Output data_edit/edits_<site>.parquet. Usage: python 39_editors.py <k> <n>"""
import os, sys, time, shutil, traceback, json
import pandas as pd, py7zr
from lxml import etree
sys.path.insert(0, os.path.dirname(__file__))
import importlib.util
spec = importlib.util.spec_from_file_location("a4", os.path.join(os.path.dirname(__file__), "04_allsites.py")); a4 = importlib.util.module_from_spec(spec); spec.loader.exec_module(a4)
BASE = a4.BASE; TMP = os.path.join(BASE, "raw_edit"); OUT = os.path.join(BASE, "data_edit"); DA = os.path.join(BASE, "data_all")
os.makedirs(TMP, exist_ok=True); os.makedirs(OUT, exist_ok=True)

def process(name, size):
    short = name.replace(".stackexchange.com", "").replace(".com", "").replace(".net", "")
    af = os.path.join(DA, f"answers_{short}.parquet"); of = os.path.join(OUT, f"edits_{short}.parquet")
    if os.path.exists(of) or not os.path.exists(af): return
    A = pd.read_parquet(af, columns=["id", "owner", "date"]); A = A[A.date >= "2019-01-01"]
    own = dict(zip(A.id.values, A.owner.values))
    p = a4.download(name, size); d = os.path.join(TMP, name + "_x")
    with py7zr.SevenZipFile(p, "r") as z: z.extract(path=d, targets=["PostHistory.xml"])
    xml = os.path.join(d, "PostHistory.xml"); rows = []
    for _, el in etree.iterparse(xml, tag="row", events=("end",), huge_tree=True):
        t = el.get("PostHistoryTypeId")
        if t in ("5", "8"):
            pid = int(el.get("PostId"))
            if pid in own:
                u = el.get("UserId"); u = int(u) if u not in (None, "") else -999
                rows.append((pid, u == own[pid] and u >= 0, el.get("CreationDate", "")))
        el.clear()
        while el.getprevious() is not None: del el.getparent()[0]
    E = pd.DataFrame(rows, columns=["id", "self", "edate"])
    g = E.groupby("id").apply(lambda x: pd.Series({"n_self": int(x.self.sum()), "n_other": int((~x.self).sum()),
                                                   "first_self": x.edate[x.self].min() if x.self.any() else None,
                                                   "first_other": x.edate[~x.self].min() if (~x.self).any() else None})) if len(E) else pd.DataFrame(columns=["n_self", "n_other", "first_self", "first_other"])
    R = A[["id", "date"]].merge(g, left_on="id", right_index=True, how="left"); R["n_self"] = R.n_self.fillna(0).astype(int); R["n_other"] = R.n_other.fillna(0).astype(int)
    R["site"] = short; R.to_parquet(of + ".tmp", index=False); os.replace(of + ".tmp", of)
    shutil.rmtree(d, ignore_errors=True); os.remove(p)
    print(time.strftime("%H:%M"), "done", short, len(R), int((R.n_self > 0).sum()), int((R.n_other > 0).sum()), flush=True)

if __name__ == "__main__":
    k, n = int(sys.argv[1]), int(sys.argv[2]); S = a4.sites()[k::n]
    print("worker", k, "sites", len(S), flush=True)
    for name, size in S:
        try: process(name, size)
        except Exception: print("FAIL", name, flush=True); traceback.print_exc()
    print("WORKER DONE", k, flush=True)
