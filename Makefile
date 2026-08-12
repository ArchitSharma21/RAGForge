.PHONY: install run test lint compile smoke verify docker

install:
	python -m pip install -r requirements-dev.txt

run:
	uvicorn app:app --host 0.0.0.0 --port 7860 --reload

test:
	pytest -q

lint:
	ruff check src tests scripts app.py

compile:
	python -m compileall -q src scripts app.py

smoke:
	python scripts/release_check.py
	pytest -q tests/test_v20_final.py tests/test_security.py tests/test_citations.py tests/test_eval_metrics.py

verify: lint test compile
	python scripts/release_check.py

docker:
	docker build -t ragforge .
	docker run --rm -p 7860:7860 --env-file .env ragforge
