# Replication Package

**Paper.** Fengming Liu, "Where Language-Model Vocabulary Rose on Stack Exchange." arXiv:2611.15172 (2026). https://arxiv.org/abs/2611.15172

**Author.** Fengming Liu, University of Chicago (fengmingliu@uchicago.edu)

A copy of the paper and its four figures is in `paper/`.

---

## 1. Overview

The paper measures how often 3,814,161 answers on 170 Stack Exchange sites (January 2019 to March 2024) use a fixed list of 56 words that language models overuse. It compares the rise of that list after the ChatGPT release with frequency-matched lists of ordinary words, and it screens 15 site traits measured before the release to find where the rise happened. The strongest predictor is the share of answers that received a body edit within 30 days of posting.

This package holds the code for every step and the derived results that the tables and figures are built from. The work has three layers.

1. **Raw data.** Public Stack Exchange data dumps on the Internet Archive. They are not included here (about 24 GB compressed for the 177 site archives of each dump). The extraction scripts download them.
2. **Intermediate data.** Per-answer counts and per-site word frequencies extracted from the dumps (about 600 MB of parquet). They hold counts only and no post text. They are attached to the GitHub release of this repository as `intermediate_data.zip`.
3. **Results.** Site-month panels, site-level tables and model estimates (`results_all/`, `results_vocab/`, about 8 MB). They are in the repository.

With the intermediate data, every table and figure can be rebuilt in about an hour on a laptop. Without it, the table and figure scripts that read only `results_vocab/` still run (see Section 5).

---

## 2. Data Availability and Provenance

### Stack Exchange data dump, April 2024

**Source.** Internet Archive item `stackexchange` (https://archive.org/details/stackexchange), the dump dated 2 April 2024. The file list used is `meta/meta_stackexchange_20240402.json`.

**Sites.** Every main site in the dump other than Stack Overflow and the meta sites. The analysis keeps the 170 sites with enough answers in each window (`results_all/exposure.csv`).

**Files used.** `Posts.xml` for answers and questions (from July 2017 on), and `PostHistory.xml` for body edits (types 5 and 8).

### Stack Exchange data dump, September 2023

**Source.** Internet Archive item `stackexchange_20230912` (https://archive.org/details/stackexchange_20230912). The file list used is `meta/meta_stackexchange_20230912.json`. It serves the check on deleted answers, since deleted answers are absent from later dumps.

### License of the data

Stack Exchange content is licensed under CC BY-SA (versions 2.5, 3.0 and 4.0, depending on the posting date). The intermediate data and the results here are derived from that content and are released under CC BY-SA 4.0 with attribution to Stack Exchange Inc. and the authors of the posts. They contain counts and word frequencies only, with no post bodies, and user ids appear only as the numeric ids already public in the dump.

---

## 3. Computational Requirements

**Software.** Python 3.13 with the packages in `requirements.txt` (pandas 2.3, numpy 2.1, statsmodels 0.14, matplotlib 3.10, pyarrow 21, lxml 6, py7zr 1.1, requests 2.32). The results were produced on Windows 11. All paths are relative to the repository root, so the code also runs on Linux and macOS.

**Hardware.** The analysis scripts need about 16 GB of RAM. Extraction streams each dump file through lxml. Each 7z archive is downloaded, parsed and then deleted, so the peak disk need is set by the largest site (math.stackexchange.com, a 3.6 GB archive that expands several times over).

**Run time.** Extraction is bound by the download speed of the Internet Archive and takes on the order of a day. The analysis scripts take about an hour in total.

**Randomness.** Every random draw uses a fixed seed set in the script (`numpy.random.default_rng`), so the permutation tests, random word lists and bootstrap draws reproduce exactly.

---

## 4. Instructions

### Setup

```
pip install -r requirements.txt
```

### Option A. From the intermediate data (recommended)

Download `intermediate_data.zip` (635 MB, SHA-256 `69EC5FFA7D20FCA9580868D7D65FE5D54341D18FC5DFED43C8839B8EA88BF42F`) from the release page and unzip it in the repository root, which creates `data_all/`, `vocab/`, `data_edit/` and `snap/`. Then run

```
python code/run_all.py
```

This runs every analysis script in order and writes a log for each into `logs/`. To start at a later script, give its number, for example `python code/run_all.py 41`.

### Option B. From the raw dumps

Run the four extraction steps first. Each takes a worker index and a worker count, so `python code/04_allsites.py 0 4` through `python code/04_allsites.py 3 4` split the sites over four processes. Each step skips sites whose output already exists, so an interrupted run can be resumed.

| Step | Script | Input | Output |
|------|--------|-------|--------|
| 1 | `04_allsites.py` | Posts.xml, 2024 dump | `data_all/answers_<site>.parquet`, `data_all/questions_<site>.parquet` |
| 2 | `07_vocab.py` | Posts.xml, 2024 dump | `vocab/vocab_<site>.parquet`, `vocab/nq_<site>.parquet`, `vocab/mk_<site>.parquet` |
| 3 | `39_editors.py` | PostHistory.xml, 2024 dump, step 1 | `data_edit/edits_<site>.parquet` |
| 4 | `11_snapshot.py` | Posts.xml, 2023 dump | `snap/snap_<site>.parquet` |

Then run `python code/run_all.py` as in Option A.

The fixed list of 56 marker words is the `EXT` set in `code/07_vocab.py`. It was set before the analysis, from the excess-vocabulary lists of Kobak et al. and Liang et al., and the other scripts read it from there.

---

## 5. Map from Scripts to the Paper

| Paper item | Script | Output in `results_vocab/` unless noted |
|------------|--------|------------------------------------------|
| Figure 1 (marker list against 1,000 matched random lists) | `42_exante_figs.py` | `paper/fig_null.pdf`, `aggnull_edit.csv`, `aggnull_edit_path.csv` |
| Figure 2 (marker change by edit tercile) | `42_exante_figs.py` | `paper/fig_edit.pdf` |
| Table 1 (site-level screen of 15 traits) | `41_exante.py`, `44_screen_exante.py`, `45_exante_extra.py`, `40_selfother.py`, `38_screen2.py` | `exante_models.csv` (rows `screen*`), `site_screen_exante.csv`, `edit_selfother_models.csv`, `site_logchange2.csv` |
| Table 2 (site-month panel) | `41_exante.py`, `46_exante_table.py`, `48_referee.py` | `exante_table.csv`, `exante_models.csv` (rows `panel_*`, `answers_*`), `referee_checks.csv` (section `B_tenure`) |
| Figure 3 (event study of the edit gradient) | `42_exante_figs.py` | `paper/fig_event.pdf`, `event_exante.csv` |
| Figure 4 (word-level ratios by edit tercile) | `43_wordfig.py` | `paper/fig_words.pdf`, `word_ratios_edit.csv` |
| Appendix table (code exposure, other outcomes, regular authors) | `16_robust.py`, `15_math.py`, `17_design.py`, `30_edit.py`, `33_within_edit_exante.py` | `robust_marker.csv`, `math_falsification.csv`, `design_checks.csv`, `edit_dose.csv`, `within_edit_exante.csv` |
| Network-wide rise of about 14 percent and its robustness | `24_aggnull.py`, `37_robust14.py` | `aggnull.csv`, `rise_robust.csv` |
| Thresholds, split halves, controls for the pre-period level, alternative reference half-years | `48_referee.py` | `referee_checks.csv` |
| Deleted answers (September 2023 dump, Id gaps) | `12_survival_exante.py`, `10_gaps.py` | `survival_*_exante.csv`, `deletion_*.csv` |
| Per-word gradients and the list without its most frequent word | `14_wordlevel.py`, `32_edit_ne.py`, `44_screen_exante.py` | `word_dose.csv`, `perm_null.csv`, `word_edit_gradient_exante.csv` |
| Numbers quoted in the text | `47_exante_desc.py`, `49_verify.py` | printed to the log |

The scripts that build the figures and Tables 1 and 2 (`41`, `42`, `43`, `44`, `46`) read only files in `results_vocab/` and `results_all/` except for the word-level parts of `42` and `43`, which also read `vocab/nq_<site>.parquet`. The scripts that build the site panels (`05`, `26`, `30`, `36`, `41`) read the intermediate data.

### Other scripts

The remaining scripts produce checks reported in passing or used to build later inputs. `05_dose.py` builds the site-month panel and the code-exposure measure (`results_all/`). `08_excess.py` tests a data-driven excess-vocabulary list, which the paper reports as failing on this corpus. `19_lenadj.py`, `21_traits.py` and `22_aggregate.py` build the length-adjusted panel, the trait battery and the aggregate series. `26_ddd.py` builds the marker score relative to matched control vocabulary (`ddd_panel.csv`). `41b_exposure_exante.py` writes the ex-ante edit share into the exposure table used by `12_survival_exante.py` and `33_within_edit_exante.py`.

---

## 6. Repository Layout

```
code/            analysis and extraction scripts, numbered in the order they were written
meta/            Internet Archive file lists for the two dumps
paper/           the paper (PDF) and its four figures
results_all/     site-month panel and site exposure measures
results_vocab/   site-level tables, model estimates and figure inputs
```

---

## 7. License

The code is released under the MIT License. The data in `results_all/`, `results_vocab/` and the release asset are released under CC BY-SA 4.0, as described in Section 2. See `LICENSE`.

---

## 8. Citation

```
@misc{liu2026stackexchange,
  author        = {Liu, Fengming},
  title         = {Where Language-Model Vocabulary Rose on {Stack Exchange}},
  year          = {2026},
  eprint        = {2611.15172},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CL},
  url           = {https://arxiv.org/abs/2611.15172}
}
```
