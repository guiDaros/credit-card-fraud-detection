# To-do (temporary)

- [ ] Retrain and re-evaluate the model. The saved `models/model.joblib` was
      trained on the raw CSV (with duplicates), but `evaluate` now uses the
      processed (deduplicated) data, so some test rows may have been seen in
      training and the metrics could be optimistic. Run:
      `python -m src.train` and then `python -m src.evaluate`.
