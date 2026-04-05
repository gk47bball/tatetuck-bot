PYTHON ?= python

.PHONY: test lint typecheck smoke

test:
	$(PYTHON) -m unittest discover -s tests -p 'test*.py'

lint:
	ruff check biopharma_agent/vnext *.py

typecheck:
	mypy biopharma_agent/vnext discord_bot.py evaluate_vnext.py operate_vnext.py readiness_vnext.py research_audit_vnext.py trade_vnext.py

smoke:
	$(PYTHON) readiness_vnext.py
	$(PYTHON) research_audit_vnext.py --top 5
