# Credit card fraud detection - common commands.
#
# Targets:
#   setup                   Install the dependencies (pip install -r requirements.txt)
#   data                    Raw CSV -> data/processed/transactions.parquet + reports/data_report.json
#   train-random            Train with the random split          -> models/model_random.joblib
#   train-temporal          Train with the temporal split        -> models/model_temporal.joblib
#   train-temporal-nohour   Temporal split, without Hour feature -> models/model_temporal_nohour.joblib
#   evaluate-random         Evaluate model_random                -> reports/metrics_random.json, pr_curve_random.png
#   evaluate-temporal       Evaluate model_temporal              -> reports/metrics_temporal.json, pr_curve_temporal.png
#   evaluate-temporal-nohour  Evaluate model_temporal_nohour     -> reports/metrics_temporal_nohour.json, ...
#   test                    Run the tests (pytest -q)
#   lint                    Check the code style (ruff check .)
#   all                     data, then the three trains, then the three evaluates (run without -j)
#
# Variables (override on the command line, e.g. `make train-temporal TRIALS=5`):
#   PYTHON   Python interpreter (default: .venv/bin/python if it exists, else python3)
#   TRIALS   Number of Optuna trials for training (default: 20)

PYTHON ?= $(if $(wildcard .venv/bin/python),.venv/bin/python,python3)
TRIALS ?= 20

.PHONY: setup data \
	train-random train-temporal train-temporal-nohour \
	evaluate-random evaluate-temporal evaluate-temporal-nohour \
	test lint all

setup:
	$(PYTHON) -m pip install -r requirements.txt

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

test:
	$(PYTHON) -m pytest -q

lint:
	$(PYTHON) -m ruff check .

all: data \
	train-random train-temporal train-temporal-nohour \
	evaluate-random evaluate-temporal evaluate-temporal-nohour
