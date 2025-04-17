FLAGS=
ifneq ($(shell whoami), bamboo)
	FLAGS+=-t
endif
DOCKER=docker run --rm -i $(FLAGS) -v $$(pwd):/raiev raiev
VERSION=$(shell python setup.py --version)

.PHONY: image
image: environment.yml requirements/requirements-all.txt requirements/requirements-dev.txt Dockerfile
	docker build -t raiev --progress=plain .

.PHONY: develop
develop: image
	$(DOCKER) bash

.PHONY: lint
lint:
	pre-commit run --all-files || true

.PHONY: wheel
wheel:
	python setup.py bdist_wheel

.PHONY: test
test:
	pytest

.PHONY: coverage
coverage:
	pytest -v --cov-report html --cov-report term --cov raiev --cov-config setup.cfg --junitxml coverage.xml tests/

.PHONY: docs
docs:
	$(MAKE) -C docs clean
	sphinx-apidoc --module-first -o docs/_API/ raiev
	cp README.md docs/_API/

	mkdir -p docs/_static
	mkdir -p docs/source
	mkdir -p docs/_build/html
	mkdir -p docs/_build/html/_static
	cp documentation_content/*.md docs/_static/
	# hacks to make images work for included .md
	cp documentation_content/_static/* docs/_static/
	cp documentation_content/_static/* docs/_build/html/_static/

	# Copy Workflows notebooks
	#cp -a WorkflowExamples docs/source

	SPHINXOPTS="-D version=$(VERSION)" $(MAKE) -C docs html
	rm -rf docs/_API  # clean up side-effects
	rm -rf docs/source  # clean up side-effects
	rm -rf docs/_static  # clean up side-effects

.PHONY: requirements
requirements:
	python requirements/requirements-util.py requirements
	CUSTOM_COMPILE_COMMAND="make requirements" python -Wi -m piptools compile -q --upgrade requirements/requirements-all.in
	python requirements/requirements-util.py requirements --no-edit | while read filename; do echo Compiling $${filename}; CUSTOM_COMPILE_COMMAND="make requirements" python -Wi -m piptools compile -q --upgrade $${filename}; done
