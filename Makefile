.PHONY: install run

install:
	@echo "Creating virtual environment..."
	python3 -m venv venv
	@echo "Activating virtual environment..."
	. venv/bin/activate
	@echo "Installing dependencies..."
	pip install -r requirements.txt

run:
	@echo "Activating virtual environment..."
	. venv/bin/activate
	@echo "Running the application..."
	python server.py
