# Diary

Chronological log of the project, newest entry last. One entry per step: what changed, the
decisions taken and why, the numbers measured, and what is left open. The rule is in `CLAUDE.md`.

## 2026-11-11: environment and torch build

- uv project, Python 3.13, src layout. Dependencies: lightning, scikit-learn, pandas, matplotlib,
  PyYAML. Dev: ruff, pytest, jupyterlab, nbconvert, ipykernel.
- torch is pinned to 2.7.1 from the CUDA 11.8 index on Linux. Why: PyTorch's release build script
  includes compute capability 3.7 (Tesla K80) only in the CUDA 11.8 wheels; the CUDA 12.x wheels
  drop it, and 2.7.1 is the newest build on the cu118 index.
- Checked on a machine without GPU: `torch.__version__` is `2.7.1+cu118` and the compiled-in arch
  flags include `sm_37`. Not yet run on a GPU. A CUDA 11.8 build needs a driver >= 450.80.02.
- pandas is 3.1.1: string columns have dtype `str`, not `object`. Column typing must use
  `pandas.api.types.is_numeric_dtype`, not `dtype == object`.
- `docs/PLAN.md` is the first plan (pass/fail framing). The project went another way, see below.

## 2026-11-11: dataset download

- `scripts/download_data.py` fetches the UCI archive into `data/raw/student-performance/`.
  The archive is nested: `student.zip` holds the CSVs; `.student.zip_old` (2012 version) is ignored.
  Separator is `;`. SHA-256 of the four extracted files is checked; a mismatch stops the script.
- Tests on a toy nested archive: only listed members extracted, checksum mismatch stops, corruption
  detected.
- The UCI API entry has no licence field. CC BY 4.0 comes from the brief and is still to be verified
  on the dataset page.

## 2026-11-11: exploration notebook 01

- `notebooks/01_explore_student_performance.ipynb` is the user's own exploration notebook. It must
  not be edited by Claude. Its last cell prototypes the excellence plots (orange/steelblue stacked
  rate bars with n) and uses a strict `>` for excellence.
- Facts measured on the maths file (n = 395): 38 students have G3 == 0 and all of them have
  absences == 0 (22% of the others have 0), so this is a recording artefact.
  Portuguese file (n = 649): 15 such students.

## 2026-11-11: direction change to excellence notebooks

Decisions, from the user:

- Target is `Excellent = G3 >= EXCELLENCE_THRESHOLD`, threshold set in each notebook (default 14).
  The user chose `>=` over the strict `>` of notebook 01.
- Rows with G3 == 0 are dropped by default (flag in `load_students`).
- Code lives in `src/data_use_ce/`; training and analysis are called from notebooks, one notebook
  per feature group. Models: decision tree and random forest first (explainable), then a Lightning
  NN, on GPU if possible.
- Evaluation, all five points selected: (1) held-out test: good students rejected and bad ones
  accepted; (2) fairness across groups; (3) how much each kind of data helps; (4) what the model
  relies on; (5) stability across random splits.

Measured (G3 >= t, G3 == 0 dropped), number of excellent students:

| Threshold t | 12 | 13 | 14 | 15 | 16 | 17 |
|---|---|---|---|---|---|---|
| Maths (n = 357) | 162 (45%) | 131 (37%) | 100 (28%) | 73 (20%) | 40 (11%) | 24 (7%) |
| Portuguese (n = 634) | 348 (55%) | 276 (44%) | 194 (31%) | 131 (21%) | 82 (13%) | 46 (7%) |

Excellent share by group at t = 14 (excellent / n):

| | Maths | Portuguese |
|---|---|---|
| address R / U | 13/78 (17%) / 87/279 (31%) | 44/187 (24%) / 150/447 (34%) |
| sex F / M | 46/185 (25%) / 54/172 (31%) | 129/376 (34%) / 65/258 (25%) |
| mother's education 4 / below 4 | 53/125 (42%) / 47/232 (20%) | 81/171 (47%) / 113/463 (24%) |
| wants higher education: no | 0/14 | 1/64 |

Consequences: with ~100 excellent students in maths, a 30% test split holds about 30 of them, and
only about 4 rural ones, so a single split is noisy and group counts per split are tiny. Repeated
splits and out-of-fold predictions are planned for the evaluation. The exploration run that was
meant to quantify this was lost when a session ended and was not repeated.

## 2026-11-11: data layer

- `data.py`: `load_students(dataset, drop_zero_final=True)`, `add_excellence(df, threshold)` (adds the
  boolean column `Excellent`, threshold from 1 to 20).
- `features.py` and `configs/feature_groups.yaml`: the five groups with the user's names.
  `load_feature_groups(columns)` fails on unknown or duplicated columns, empty groups, and on `G3` or
  `Excellent` used as a feature. Every predictor is in exactly one group (tested on the shipped file).
- Result on the real files: maths 395 -> 357 rows after dropping zeros, 100 excellent at t = 14.

## 2026-11-11: plot helpers and the school notebook

- `plots.py`: `plot_excellence_rate` (the user's orange/steelblue bars: excellent share per value,
  n labelled, dashed line = overall excellent share), `plot_feature_histogram` (side-by-side
  shares within each class, for numeric columns with more than 10 distinct values),
  `plot_scatter` (coloured by class, seeded jitter), `plot_feature_grid` (one panel per feature,
  chooses bars or histogram). English labels for now; Italian labels come with the slide figures.
- `notebooks/02_group_school.ipynb`: parameters (`DATASET`, `EXCELLENCE_THRESHOLD`, `GROUP`), feature
  table, grid of plots, closer look, empty cell. Shipped without outputs; run headless once: no
  errors, 4 plots. `tests/conftest.py` sets the Agg backend and closes figures.
- What the school plots show (maths, t = 14, overall 28% excellent): no past failure 33% (n = 294)
  vs 5-8% with one or two (n = 52); extra school support 6% (n = 50) vs 32% without; wants higher
  education: 0% for the 14 who do not; study time 3 and 4: 44% and 42%, vs 27% and 21% for study time 1 and 2; travel time 1:
  32% vs 16% for 3; school chosen for reputation 36% vs 23-27%. `paid` and `activities` are close to
  the average. Support and failures mark weaker students, so a model will use them.
- 21 tests pass. `ruff check` is clean on `src`, `tests`, `scripts` and the new notebook.

Open:

- Notebook 01's excellence cell still uses `>`; the package uses `>=`.
- `ruff check .` flags long lines (E501) in notebook 01; decide whether to exclude `notebooks/`
  (note that `ruff format .` would rewrite the user's cells).
- `CLAUDE.md` still has lines from another project (cu126 index, slurm queue, Hydra rule, jets/s).
- Next: split and preprocessing, decision tree and random forest, rejected/accepted counts with a
  random-selection baseline, then repeated splits.
