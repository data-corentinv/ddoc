# CONFIGURATION
SOURCE_DIR = ddoc

COVERAGE_OUTPUT = $(REPORTS_DIR)/coverage
COVERAGE_OPTIONS = --cov-config coverage/.coveragerc --cov-report term --cov-report html

# .PHONY is used to distinguish between a task and an existing folder
.PHONY: doc tests coverage word excel install

# COMMANDS
install: 
	python3 -m venv .venv
	. .venv/bin/activate && \
		pip install -r requirements.txt --no-cache-dir
	
word:
	python ddoc/main.py --type-report='word' --out-directory='$(out_directory)' --metadata-location='$(metadata_location)' --data-location='$(data_location)' --addons='$(addons)'

excel:
	python ddoc/main.py --type-report='excel' --out-directory='$(out_directory)' --data-location='$(data_location)'

doc:
	echo TODO

tests: 
	docker-compose run app pytest -s tests/unit_tests/

coverage: # not check yet
	py.test $(COVERAGE_OPTIONS) --cov=$(SOURCE_DIR) tests/unit_tests/ | tee coverage/coverage.txt
	mv -f .coverage coverage/.coverage  # don't know how else to move it
	mkdir -p $(COVERAGE_OUTPUT)
	cp -r coverage/htmlcov/* $(COVERAGE_OUTPUT)

clean: clean-dev
	rm -rf ddoc.egg-info
	find -iname "__pycache__" -delete

clean-dev: 
	rm -rf .venv
	find -iname "*.pyc" -delete

