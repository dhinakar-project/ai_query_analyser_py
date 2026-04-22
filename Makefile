.PHONY: install run test evals clean

install:
	pip install -r requirements.txt

run:
	streamlit run app.py

test:
	pytest tests/ -v

evals:
	python -m evals.cli

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true

help:
	@echo "Available commands:"
	@echo "  make install  - Install all dependencies"
	@echo "  make run      - Start the Streamlit app"
	@echo "  make test     - Run unit tests"
	@echo "  make evals    - Run evaluation benchmark"
	@echo "  make clean    - Remove cached files"
