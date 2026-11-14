# Credit card fraud detection - common commands.
#
# Targets:
#   setup                   Install the dependencies (requirements.txt + requirements-dev.txt)
#   setup-lock              Install the exact versions used for the results (requirements-lock.txt)
#   data                    Raw CSV -> data/processed/transactions.parquet + reports/data_report.json
#   train-random            Train with the random split          -> models/model_random.joblib
#   train-temporal          Train with the temporal split        -> models/model_temporal.joblib
#   train-temporal-nohour   Temporal split, without Hour feature -> models/model_temporal_nohour.joblib
#   evaluate-random         Evaluate model_random                -> reports/metrics_random.json, pr_curve_random.png
#   evaluate-temporal       Evaluate model_temporal              -> reports/metrics_temporal.json, pr_curve_temporal.png
#   evaluate-temporal-nohour  Evaluate model_temporal_nohour     -> reports/metrics_temporal_nohour.json, ...
#   baseline-temporal       Logistic regression baseline, temporal split -> reports/metrics_baseline_temporal.json
#   compare-hour            With vs without Hour, temporal CV on train only -> reports/compare_hour.json
#   test                    Run the tests (pytest -q)
#   lint                    Check the code style (ruff check .)
#   notebooks               Execute the notebooks and save their outputs in place
#   publish                 Upload one configuration to the Hugging Face Hub (needs REPO=...)
#   all                     data, then the three trains, then the three evaluates (run without -j)
#
# Variables (override on the command line, e.g. `make train-temporal TRIALS=5`):
#   PYTHON   Python interpreter (default: .venv/bin/python if it exists, else python3)
#   TRIALS   Number of Optuna trials for training (default: 20)
#   REPO     Hugging Face repository id for `publish`, e.g. your-username/credit-card-fraud-rf
#   SPLIT    Configuration to publish: random or temporal (default: temporal)

PYTHON ?= $(if $(wildcard .venv/bin/python),.venv/bin/python,python3)
TRIALS ?= 20
SPLIT ?= temporal

.PHONY: setup setup-lock data \
	train-random train-temporal train-temporal-nohour \
	evaluate-random evaluate-temporal evaluate-temporal-nohour \
	baseline-temporal compare-hour \
	test lint notebooks publish all

setup:
	$(PYTHON) -m pip install -r requirements.txt -r requirements-dev.txt

setup-lock:
	$(PYTHON) -m pip install -r requirements-lock.txt -r requirements-dev.txt

data:
	$(PYTHON) -m src.make_dataset

train-random:
	$(PYTHON) -m src.train --trials $(TRIALS) --split random

train-temporal:
	$(PYTHON) -m src.train --trials $(TRIALS) --split temporal

train-temporal-nohour:
	$(PYTHON) -m src.train --trials $(TRIALS) --split temporal --no-hour

evaluate-random:
	$(PYTHON) -m src.evaluate --split random

evaluate-temporal:
	$(PYTHON) -m src.evaluate --split temporal

evaluate-temporal-nohour:
	$(PYTHON) -m src.evaluate --split temporal --no-hour

baseline-temporal:
	$(PYTHON) -m src.baseline

compare-hour:
	$(PYTHON) -m scripts.compare_hour

test:
	$(PYTHON) -m pytest -q

lint:
	$(PYTHON) -m ruff check .

# nbconvert runs each notebook from its own folder, as the notebooks expect.
notebooks:
	$(PYTHON) -m jupyter nbconvert --to notebook --execute --inplace notebooks/*.ipynb

publish:
	@test -n "$(REPO)" || (echo "Usage: make publish REPO=your-username/repo-name [SPLIT=temporal]"; exit 1)
	$(PYTHON) -m src.publish --repo $(REPO) --split $(SPLIT)

all: data \
	train-random train-temporal train-temporal-nohour \
	evaluate-random evaluate-temporal evaluate-temporal-nohour
