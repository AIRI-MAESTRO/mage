.PHONY: install sync test test-cov lint format typecheck clean package all example-fast example-deep example-web example-memory example-full examples

# --- Setup ---

install: ## Install project + dev deps via uv
	uv sync --group dev

sync: ## Sync all deps (including optional) via uv
	uv sync --group dev --all-extras

# --- Quality ---

test: ## Run tests
	uv run pytest tests/ -v --tb=short

test-cov: ## Run tests with coverage report
	uv run pytest tests/ -v --cov=mmar_mage --cov-report=term-missing --tb=short

lint: ## Run ruff linter
	uv run ruff check mmar_mage/ tests/

format: ## Auto-format code with ruff
	uv run ruff format mmar_mage/ tests/

typecheck: ## Run mypy type checker
	uv run mypy mmar_mage/

# --- Utilities ---

clean: ## Remove build artifacts and caches
	rm -rf __pycache__ .pytest_cache .ruff_cache .mypy_cache dist build *.egg-info .venv
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

package: ## Build a CLEAN source archive — git-tracked files only (no .venv/__MACOSX/caches)
	@mkdir -p dist
	git archive --format=tar.gz --prefix=mmar-mage/ -o dist/mmar-mage-source.tar.gz HEAD
	@echo "Wrote dist/mmar-mage-source.tar.gz — tracked files only (no .venv / __MACOSX / caches)"

all: lint typecheck test ## Run lint + typecheck + tests

# --- Examples ---

example-fast: ## Run fast mode example
	uv run python examples/fast_mode/run.py

example-deep: ## Run deep mode (local) example
	uv run python examples/deep_mode_local/run.py

example-web: ## Run deep mode + web research example
	uv run python examples/deep_mode_web_research/run.py

example-memory: ## Run deep mode + memory research example
	uv run python examples/deep_mode_memory_research/run.py

example-full: ## Run full deep mode example
	uv run python examples/deep_mode_full/run.py

examples: example-fast example-deep example-web example-memory example-full ## Run all examples

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-15s\033[0m %s\n", $$1, $$2}'
