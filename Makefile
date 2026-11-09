.PHONY: install test demo serve clean

install:
	python -m pip install -r requirements.txt

test:
	python -m pytest -q

demo:
	python scripts/demo.py

serve:
	python -m knowledge_assistant serve --docs data/sample_docs

clean:
	rm -rf output .pytest_cache
