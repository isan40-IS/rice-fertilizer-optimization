install:
	pip install -e .

dev:
	pip install -r requirements-dev.txt

test:
	pytest

lint:
	ruff check src tests scripts

preprocess:
	python scripts/preprocess.py

train:
	python scripts/train_models.py

full:
	python scripts/run_full_experiment.py
