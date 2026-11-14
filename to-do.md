# To-do

Next steps, in order. The code is ahead of the saved model and the reports: until
step 3 is done, the README numbers describe the old model.

The README rewrite (step 5) must use only numbers from `reports/*.json`. Steps 2-4
exist so that every number it needs is in those files first.

## 1. Data report (no training, a few seconds)

- [ ] `make data`, then commit `reports/data_report.json` (provenance with
      `git_dirty: false`; the data itself does not change).

## 2. Code still missing for the README (no training)

- [ ] **Logistic regression baseline on the temporal split.** Nothing writes it to
      `reports/` today (`train_baseline` is only used in notebook 02, on the random
      split). Add a script or make target that fits feature step + StandardScaler +
      LogisticRegression on the temporal training set, picks its threshold with the
      same temporal out-of-fold F1, and writes `reports/metrics_baseline_temporal.json`
      with PR-AUC, precision, recall, F1, both bootstrap intervals and provenance.
- [ ] **compare_hour.py must save its result.** It only prints. Make it write
      `reports/compare_hour.json` (per-fold PR-AUC with and without Hour, mean, std,
      paired difference, parameters used, n_splits, provenance).
- [ ] **CV details for the cv_pr_auc explanation.** Record in the training
      provenance the number of CV folds, the rows and frauds that got an out-of-fold
      prediction, and their fraud rate, so the README can say from the files why
      `cv_pr_auc` and the test PR-AUC are not directly comparable.
- [ ] Optional: record the test period (first and last hour of `Time`) in
      `metrics_*.json`, so "the temporal test is a single evening (hours 16-23)" can
      also come from a file (today only `n_blocks` = number of hours would).

## 3. Retrain and re-evaluate (about 30 minutes)

- [ ] Run with a clean tree (commit code first; changes in `reports/` are allowed):

      nohup sh -c 'make train-temporal && make evaluate-temporal && make notebooks' > train.log 2>&1 &

- [ ] Run the baseline and `python -m scripts.compare_hour` (step 2 outputs).
- [ ] Check that `reports/metrics_temporal.json` has `bootstrap_ci`,
      `block_bootstrap_ci`, `cv_pr_auc` and `training_provenance` (which holds the
      exact scikit-learn version of the pickle), and that the new
      `models/model_temporal.joblib` loads outside the repository.
- [ ] If compare_hour shows no benefit from Hour, decide whether to drop it
      (retrain with `--no-hour`) before publishing.
- [ ] Commit `reports/` and the executed notebooks. Delete `train.log`.

Results will differ from the current PR-AUC 0.817: `class_weight` is now tuned, so
the Optuna search follows a different path.

## 4. Publish the model to Hugging Face (before the final README)

- [ ] Confirm the Hub repository id (the README currently assumes
      `guilhermedaros/credit-card-fraud-rf`; not verified).
- [ ] `make publish REPO=<user>/credit-card-fraud-rf`.
- [ ] The README usage example (`hf_hub_download` + `joblib.load` + predict on a raw
      DataFrame) can only be tested for real once the model is on the Hub; test it
      from a folder outside the repository.

## 5. README / model card rewrite

Prompt ready to use once steps 1-4 are done. It covers: results table with both
intervals, baseline row, `cv_pr_auc` with an honest explanation, YAML with
`pipeline_tag` and without `datasets:`, exact scikit-learn version and pickle
warning, tested usage example, absolute GitHub links to the notebooks, link to the
Hub page, complete limitations, and the Hour and class_weight decisions.

- [ ] Run the README prompt. Then `make publish` again so the Hub card matches.
- [ ] Also mention in the README: Python 3.12, `requirements-dev.txt` /
      `requirements-lock.txt`, and the make targets `setup-lock`, `notebooks`,
      `publish` (not in the prompt).

## 6. Notebook text

- [ ] Notebook 02, `cell-11`: remove "balanced_subsample is kept in the final model"
      (class_weight is now tuned).
- [ ] Notebook 03: re-check the hard-coded numbers in the markdown of sections 3-5
      and the summary against the new outputs.

## 7. Optional comparisons (training set only for decisions)

- [ ] `make train-random evaluate-random` and the no-`Hour` configuration, then
      update the comparison in notebook 03 and the README.

## Smaller cleanups

- [ ] Use `logging` instead of `print` in train, evaluate, make_dataset and publish.
- [ ] Decide whether commits should use the GitHub noreply e-mail before making the
      repository public.
