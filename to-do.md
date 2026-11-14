# To-do

Next steps, in order. The code is ahead of the saved model and the reports: until
step 2 is done, the README numbers describe the old model.

## 1. Data report (no training, a few seconds)

- [ ] `make data`, then commit `reports/data_report.json` (provenance with
      `git_dirty: false`; the data itself does not change).

## 2. Retrain and re-evaluate (about 30 minutes)

- [ ] Run with a clean tree (commit code first; changes in `reports/` are allowed):

      nohup sh -c 'make train-temporal && make evaluate-temporal && make notebooks' > train.log 2>&1 &

- [ ] Check that the new `models/model_temporal.joblib` loads outside the repository
      (no `src` needed) and that `reports/metrics_temporal.json` has
      `bootstrap_ci`, `block_bootstrap_ci`, `cv_pr_auc` and `training_provenance`.
- [ ] Commit `reports/` and the executed notebooks. Delete `train.log`.

Results will differ from the current PR-AUC 0.817: `class_weight` is now tuned, so
the Optuna search follows a different path.

## 3. Update the text with the new numbers

- [ ] README: results table with both intervals (row and one-hour block bootstrap)
      and `cv_pr_auc` next to the test PR-AUC; say the temporal test spans only
      8 hours.
- [ ] README: `class_weight` is tuned (not fixed to `balanced_subsample`); add it to
      the search space description.
- [ ] README: "How to use the model" no longer needs the `src` package.
- [ ] README: Python 3.12, `requirements-dev.txt` / `requirements-lock.txt`, and the
      new make targets (`setup-lock`, `notebooks`, `publish`).
- [ ] README limitations: `Hour` is relative to the first transaction of the 2013
      dump (new data needs `Time` in the same frame); the threshold comes from
      models trained on part of the training set; the intervals do not include
      training or threshold variance.
- [ ] Notebook 02, `cell-11`: remove "balanced_subsample is kept in the final model".
- [ ] Notebook 03: re-check the hard-coded numbers in the markdown of sections 3-5
      and the summary.

## 4. Optional comparisons (training set only)

- [ ] `python -m scripts.compare_hour` to decide about `Hour` with temporal CV
      (never with the test set).
- [ ] `make train-random evaluate-random` and the no-`Hour` configuration, then
      update the comparison in notebook 03 and the README.

## 5. Hugging Face (only after steps 2-3)

- [ ] Model card YAML: add `pipeline_tag: tabular-classification`; replace or remove
      `datasets: mlg-ulb/creditcardfraud` (a Kaggle id, not a Hub dataset).
- [ ] README: link to the Hugging Face model page; use an absolute GitHub URL for
      the notebook link (relative links break on the Hub).
- [ ] `make publish REPO=<user>/credit-card-fraud-rf`.

## Smaller cleanups

- [ ] Use `logging` instead of `print` in train, evaluate, make_dataset and publish.
- [ ] Decide whether commits should use the GitHub noreply e-mail before making the
      repository public.
