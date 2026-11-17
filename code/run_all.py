"""Run the analysis scripts in dependency order (intermediate data in data_all/, vocab/, data_edit/, snap/ must exist).
Usage: python code/run_all.py            run everything from 05 on
       python code/run_all.py 41         start at the first script whose number is >= 41
Extraction (04, 07, 11, 39) is not run here. See README."""
import subprocess, sys, time, pathlib

ORDER = ["05_dose", "08_excess", "09_wordtraj", "10_gaps", "14_wordlevel", "15_math", "16_robust", "17_design",
         "18_joint", "19_lenadj", "20_jointevent", "21_traits", "22_aggregate", "23_tercile", "24_aggnull", "26_ddd",
         "27_siteratio", "28_boot", "29_screen", "30_edit", "31_interact", "32_edit_ne", "36_unedited", "37_robust14",
         "38_screen2", "40_selfother", "41_exante", "41b_exposure_exante", "12_survival_exante", "33_within_edit_exante",
         "42_exante_figs", "43_wordfig", "44_screen_exante", "45_exante_extra", "46_exante_table", "47_exante_desc",
         "48_referee", "49_verify"]

if __name__ == "__main__":
    here = pathlib.Path(__file__).resolve().parent
    start = sys.argv[1] if len(sys.argv) > 1 else None
    seq = ORDER if start is None else ORDER[next(k for k, s in enumerate(ORDER) if s.startswith(start)):]
    log = here.parent / "logs"; log.mkdir(exist_ok=True)
    for s in seq:
        t = time.time()
        with open(log / f"{s}.log", "w", encoding="utf8") as f:
            r = subprocess.run([sys.executable, str(here / f"{s}.py")], stdout=f, stderr=subprocess.STDOUT)
        print(f"{s:28s} exit={r.returncode} {time.time() - t:6.0f}s", flush=True)
        if r.returncode != 0:
            sys.exit(f"{s} failed, see logs/{s}.log")
