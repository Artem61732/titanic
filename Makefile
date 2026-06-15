.PHONY: install run run-quick run-dl

install:
	pip install -r requirements.txt

run:
	python main.py

run-quick:
	python main.py --quick

run-dl:
	python main.py --pipeline dl
