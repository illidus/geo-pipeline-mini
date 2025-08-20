.PHONY: setup test run clean help

# Default target
help:
	@echo "Geo Pipeline Mini - Available commands:"
	@echo "  setup     - Create virtual environment and install dependencies"
	@echo "  test      - Run unit tests"
	@echo "  run       - Run pipeline with demo data"
	@echo "  plots     - Generate visualization plots"
	@echo "  clean     - Clean up generated files"
	@echo "  help      - Show this help message"

setup:
	@echo "Setting up development environment..."
	python -m venv .venv
	.venv/bin/pip install --upgrade pip
	.venv/bin/pip install -r requirements.txt
	@echo "Setup complete. Activate with: source .venv/bin/activate"

test:
	@echo "Running unit tests..."
	python -m pytest tests/ -v

run:
	@echo "Running geo pipeline with demo data..."
	@mkdir -p outputs assets
	python -m src.pipeline.soil_pipeline run_pipeline --data data/demo --out outputs
	@echo "Pipeline completed. Results in outputs/"

plots:
	@echo "Generating visualization plots..."
	@mkdir -p assets
	python -m src.pipeline.plots
	@echo "Plots saved to assets/"

clean:
	@echo "Cleaning up..."
	rm -rf outputs/* assets/*.png
	rm -rf __pycache__ src/__pycache__ src/pipeline/__pycache__ tests/__pycache__
	rm -rf .pytest_cache
	@echo "Cleanup complete."