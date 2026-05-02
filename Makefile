.PHONY: bronze silver gold tableau all test quality geo-reference clean

PYTHON ?= python

bronze:
	$(PYTHON) pipeline/run_bronze.py

silver: bronze
	$(PYTHON) pipeline/run_silver.py

gold: silver
	$(PYTHON) pipeline/run_gold.py

tableau: gold
	$(PYTHON) pipeline/run_tableau.py

all: bronze silver gold tableau

test:
	$(PYTHON) -m pytest tests/ -v

quality: test
	$(PYTHON) -c "from src.medallion.metadata import validate_layer_manifests; validate_layer_manifests()"

geo-reference:
	$(PYTHON) pipeline/fetch_postal_centroids.py

clean:
	rm -rf data/bronze data/silver data/gold data/hyper data/medallion_metadata
