"""Build results_vocab/exposure_plus_exante.csv: exposure_plus.csv with r_edit replaced by the ex-ante 30-day edit share rank (r_ea30) from 41_exante.py. Used by 12_survival_exante.py and 33_within_edit_exante.py."""
import pandas as pd
R = str(__import__("pathlib").Path(__file__).resolve().parents[1]) + "/results_vocab"
E = pd.read_csv(R + "/exposure_plus.csv"); S = pd.read_csv(R + "/exante_edit.csv")[["site", "r_ea30"]]
E = E.merge(S, on="site", how="left"); E["r_edit"] = E.pop("r_ea30")
E.to_csv(R + "/exposure_plus_exante.csv", index=False)
