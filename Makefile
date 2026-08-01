.PHONY: init

init:
	@echo "Starting backend API on http://localhost:8000 ..."
	@(cd backend && uv run uvicorn main:app --reload --port 8000 &)
	@sleep 1
	@echo "Opening frontend..."
	@open frontend/index.html
