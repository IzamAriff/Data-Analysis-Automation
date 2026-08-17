.PHONY: install backend frontend dev test docker-up docker-down clean

install:
	pip install -r requirements.txt
	cd frontend && npm ci

backend:
	uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 --reload

frontend:
	cd frontend && npm run dev

# Run both in parallel (requires tmux or two terminals normally)
dev:
	@echo "Run 'make backend' in one terminal and 'make frontend' in another"
	@echo "Or use docker-compose up"

test:
	pytest tests/ -v
	cd frontend && npm run build

docker-up:
	docker-compose up --build

docker-down:
	docker-compose down

docker-legacy:
	docker-compose --profile legacy up --build

clean:
	find . -type d -name __pycache__ -prune -exec rm -rf {} \;
	rm -rf frontend/dist frontend/node_modules
