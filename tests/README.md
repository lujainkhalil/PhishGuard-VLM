# Tests

- **unit/** — Pure functions, single components (feeds, preprocessing, metrics, aggregator).
- **integration/** — Pipeline stages with real or fixture data (crawler + preprocess, model forward).
- **e2e/** — Full API test (e.g. mock crawl + model) or small real dataset.

PyTest; fixtures for small datasets and mock Wikidata/Playwright where needed.
