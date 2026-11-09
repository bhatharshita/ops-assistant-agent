# Shortcuts. Run `make help` to list them.
.PHONY: help up down db install ingest api ui test test-db eval lint mcp

help:  ## list targets
	@grep -E '^[a-z-]+:.*##' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  %-10s %s\n", $$1, $$2}'

up:  ## build and start the whole stack (db, ingest, api, ui)
	docker compose up --build

down:  ## stop the stack and delete the database volume
	docker compose down -v

db:  ## start only PostgreSQL + pgvector (for local development)
	docker compose up -d db

install:  ## install backend (editable, with dev tools) and frontend dependencies
	cd backend && pip install -e ".[dev]"
	cd frontend && npm install

ingest:  ## load the sample issues and logs into Postgres
	cd backend && python -m ops_assistant.ingest

api:  ## run FastAPI with auto-reload on :8000
	cd backend && uvicorn ops_assistant.api.main:app --reload

ui:  ## run the React dev server on :5173
	cd frontend && npm run dev

test:  ## unit tests (backend + frontend)
	cd backend && pytest
	cd frontend && npm test

test-db:  ## backend tests including pgvector integration tests (uses database ops_test)
	cd backend && TEST_DATABASE_URL=postgresql://ops:ops@localhost:5432/ops_test pytest

eval:  ## run the evaluation suite with the current settings
	cd backend && python -m evals.run_eval

lint:  ## ruff + TypeScript type check
	cd backend && ruff check .
	cd frontend && npm run typecheck

mcp:  ## start the MCP server on stdio
	cd backend && python -m ops_assistant.mcp_server
