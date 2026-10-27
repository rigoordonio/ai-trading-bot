# Local CI. Paper trading must not run if any step fails (make stops at the first error).
.PHONY: lint type unit integration coverage ci paper

lint:
	ruff check src tests scripts
type:
	mypy src
unit:
	pytest -m "not integration and not slow"
integration:
	pytest -m integration
coverage:
	pytest --cov=src --cov-report=term-missing --cov-report=html
	coverage report --fail-under=85 --include="src/risk/*,src/execution/orders.py,src/execution/order_manager.py,src/execution/simulator.py,src/execution/paper_broker.py,src/portfolio/portfolio.py,src/safety/*,src/data/validator.py"
ci: lint type unit integration coverage
paper: ci
	TRADING_MODE=PAPER python scripts/paper_trade.py
