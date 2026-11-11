# Project plan (draft)

Civic-education lesson (≈2 h, first-year Italian high-school students, 14–15 y.o.) on how a
school could misuse personal data to select students. A model predicts who passes the year
from the students' background; students then check fairness with simple statistics.

Students do not code. The repo produces trained models, reproducible results, slide figures,
and test tables with answer keys.

## Status

- [x] Environment: `pyproject.toml` + `uv.lock`, src layout, torch 2.7.1 CUDA 11.8 build on Linux (K80 compatible), ruff, pytest.
- [x] Download pipeline: `scripts/download_data.py` → `data/raw/student-performance/`.
- [x] Exploration notebook: `notebooks/01_explore_student_performance.ipynb`.
- [ ] Config files (feature groups, experiments, decision rule).
- [ ] Models and cross-validation.
- [ ] Decision rules, fairness metrics, bootstrap intervals.
- [ ] Lightning MLP.
- [ ] `run` end to end, metrics summary per experiment.
- [ ] Tree plot and depth sweep.
- [ ] Proxy analysis, contingency tables, quasi-identifier uniqueness.
- [ ] Slide figures (Italian labels).
- [ ] Test tables and answer keys (Markdown and LaTeX).
- [ ] README, CLAUDE.md, `reproduce` command.

## Data

- Source: UCI *Student Performance*, https://archive.ics.uci.edu/dataset/320/student+performance
  (DOI 10.24432/C5TG7T). Licence stated as CC BY 4.0 in the brief; the UCI API has no
  licence field, so verify it on the dataset page.
- The archive is nested: `student.zip` contains the CSVs; `.student.zip_old` is ignored.
  CSV separator is `;`.
- Files: `student-mat.csv` (395 students, default), `student-por.csv` (649 students).
- Target: pass = `G3 >= 10`. 38 maths and 15 Portuguese students have `G3 == 0`, and all of
  them have `absences == 0`. Handling is configurable (see open decisions).

## Feature groups (planned config `configs/features.yaml`)

| group | columns |
|---|---|
| prior_grades | G1, G2 |
| school | failures, absences, studytime, schoolsup, paid, activities, higher, reason, traveltime |
| family_socioeconomic | Medu, Fedu, Mjob, Fjob, famsize, Pstatus, address, internet, famsup, guardian, nursery |
| demographic | sex, age, school |
| lifestyle_health | Dalc, Walc, health, romantic, goout, freetime, famrel |

Every non-grade column is in exactly one group (checked).

## Planned structure

```
configs/     features.yaml · base.yaml · experiments.yaml
scripts/     download_data.py
notebooks/   01_explore_student_performance.ipynb
src/data_use_ce/
  cli.py  config.py  data.py  features.py  models.py  mlp.py  cv.py
  decisions.py  fairness.py  proxy.py  privacy.py  tables.py  plots.py  labels_it.py
tests/       fairness, decisions, no-leakage, tables
data/        raw/ (gitignored)
outputs/     <dataset>/<experiment>/ (gitignored)
```

## Models

- Majority-class baseline.
- Logistic regression (reference).
- Shallow decision tree (depth 2–4) for the plot, plus a depth sweep (train vs validation).
- Small MLP in Lightning: one-hot categoricals, standardised numerics, configurable hidden
  layers, dropout, weight decay, early stopping (validation split taken from the training fold).
- Every preprocessing step is fitted inside the training fold (sklearn Pipeline).

## Evaluation

- Repeated stratified 5-fold CV, 10 repeats by default. Out-of-fold predictions saved for every
  student (row id, repeat, fold, score, decision, true label, protected attributes).
- Performance: accuracy, balanced accuracy, ROC AUC (mean ± std), always next to the baseline.
- Decision rule: admit the top X% by score (proposed X = 60%), or threshold 0.5.
  Proposed: rank **within each held-out fold**, since each fold acts as a new applicant group
  and scores from different fold models are not on a common scale.
- Fairness per protected attribute: selection rate, disparate impact ratio (flag < 0.8 or
  > 1.25), false negative rate among students who passed, equal opportunity difference,
  per-group confusion matrices, bootstrap confidence intervals (2,000 resamples).
- Proxy analysis: predict each protected attribute from the features left after removing the
  sensitive groups; contingency tables address × traveltime, address × internet.
- Optional: quasi-identifier uniqueness (school, sex, age, address, Mjob, Fjob). Already checked:
  29% of maths students are unique on these six columns.

## Open decisions (my recommendation first)

1. **G3 == 0:** drop by default, with "keep as fail" as an option. Maths would have 357 students
   and a 74% pass rate.
2. **Experiments:** `all_with_grades`, `all`, `no_protected_columns` (drop only address, Medu,
   Fedu, sex: the naive fix), `school_only` (sensitive data removed), `family_only`.
   The brief's "remove the sensitive data" is the same as `school_only`, so I'd add the
   naive-fix experiment.
3. **Group name clash:** the group `school` and the column `school` (GP/MS). Proposed rename
   of the group to `school_record`.
4. **`absences`:** recorded in the same year as G3, so it's not available at admission in the
   strict sense. Keep it, and present it as "this year's absences" to students.
5. **Proxies inside `school`:** `paid`, `reason`, `traveltime` carry socio-economic signal.
   Kept on purpose: useful for the lesson.
6. **Reference group per attribute:** set explicitly in the config (sex differs by dataset:
   boys pass more in maths, girls in Portuguese). Disparate impact = other / reference.
7. **Mother's education threshold:** Medu == 4 (degree) vs the rest. Medu ≥ 3 shows almost no
   gap in maths (75% vs 73%).
8. **X in "top X%":** 60%. Any X below the 74% pass rate rejects some students who would pass.
9. **Dataset for the lesson:** maths is the default, but the gaps are small (rural 70% vs urban
   75%, 78 rural students). Portuguese shows larger gaps. I'll run both and compare before
   making the slides.
10. **Decision rule:** within-fold ranking (proposed) vs global ranking. Ties are broken with a
    seeded random draw.

## Working agreements

- Config-driven experiments, no edits to code to change an experiment.
- All random seeds fixed and derived from one base seed.
- Code, comments and docs in English. Figure labels and student-facing tables in Italian.
- Commits only when you ask (see CLAUDE.md).
