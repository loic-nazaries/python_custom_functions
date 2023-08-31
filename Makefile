.PHONY: all

all: create_directories create_files initialise_git create_gitignore install_poetry initialise_poetry activate_venv install_precommit install_packages setup_environment update_pip install_dependencies update_poetry install_dependencies_requirements generate_docs create_dataset clean_data full_process_clean_data run_eda clean_repo

# Below commands allow for setting up a repository structure and a new virtual environment using the Poetry Python library

# Define the variable that contain the list of directories to create
DIRECTORIES := src figures output docs prod tests

create_directories:
	@echo "Creating Default Folders..."
	$(foreach dir,$(DIRECTORIES),\
		if not exist $(dir) (mkdir $(dir)) &&) :

create_files: create_directories
	@echo "Creating Default Files..."
	@echo """Main File""" > main.py
	if not exist scripts mkdir scripts
	echo. > src\__init__.py
	echo. > src\.gitkeep
	echo. > figures\.gitkeep
	echo. > output\.gitkeep
	echo. > docs\.gitkeep
	echo. > prod\.gitkeep
	echo. > tests\__init__.py
	echo. > tests\.gitkeep
	echo. > .env
	echo. > README.md

initialise_git:
	@echo "Initialising git..."
	git init

create_gitignore: initialise_git
	@echo "Creating .gitignore..."
	@echo "# Ignore compiled files and directories" > .gitignore
	@echo __pycache__/ >> .gitignore
	@echo *.pyc >> .gitignore

install_poetry:
	@echo "Installing poetry..."
	pip install poetry

initialise_poetry:
	@echo "Initialising poetry..."
	poetry init

# Activation of venv using poetry module
activate_venv: install_poetry, initialise_poetry
	@echo "Activating Virtual Environment..."
	poetry shell

# Activation of venv using 'venv' module NOT working
# Define the variable for the name of the virtual environment
VENV := venv
create_activate_venv:
	@echo "Creating Virtual Environment..."
	python -m venv $(VENV)
	@echo "Activating Virtual Environment..."
	$(VENV)/Scripts/activate

# Since above not working, try alternate version below
$(VENV)/Scripts/activate: requirements.txt
	pyhon -m venv $(VENV)
	./$(VENV)/Scripts/pip install -r requirements.txt
	@echo "Updating pip package..."
	pip install --upgrade pip
# 'venv' is a shortcut target
venv: $(VENV)/Scripts/activate

install_precommit:
	@echo "Installing pre-commit hooks..."
	poetry add pre-commit
	poetry run pre-commit install
	poetry run pre-commit sample-config >> .pre-commit-config.yaml

install_packages:
	@echo "Installing Default Packages..."
	poetry add black flake8 ruff pylint pyupgrade add-trailing-comma dead interrogate refurb sourcery mypy

setup_environment: create_directories create_files initialise_git create_gitignore install_poetry initialise_poetry activate_venv install_precommit install_packages create_directories create_files

####################################################################

update_pip:
	@echo "Updating pip packages..."
	pip install --upgrade pip

# Install dependencies from a 'pyproject.toml' file
install_dependencies:
	@echo "Installing Dependencies from pyproject.toml..."
	poetry install

update_poetry:
	@echo "Updating poetry..."
	poetry self update

# Install dependencies from a 'requirements.txt' file
install_dependencies_requirements: create_activate_venv
	@echo "Installing Dependencies from requirements.txt..."
	./$(VENV)/Scripts/pip install -r requirements.txt

# Code below must be modified by hand to select specific folders as it cannot be done automatically
generate_docs:
	@echo "Generating Documentation..."
	pdoc3 --html --output-dir ./docs ./scripts --force

####################################################################

# Commands specific to this repository

create_dataset:
	@echo "Creating the Dataset..."
	python scripts/create_dataset/create_dataset.py

process_data:
	@echo "Creating the Dataset..."
	python scripts/cluster_manipulation/prepare_data.py

clean_data:
	@echo "Cleaning the Data..."
	python scripts/clean_data/clean_data.py

full_process_clean_data: create_dataset, process_data
	clean_data

run_eda:
	@echo "Running EDA..."
	python scripts/exploratory_data_analysis/run_exploratory_data_analysis.ipynb

####################################################################

# Below commands are NOT implemented yet (they are just examples)

download_data:
	@echo "Downloading data..."
	wget https://gist.githubusercontent.com/khuyentran1401/a1abde0a7d27d31c7dd08f34a2c29d8f/raw/da2b0f2c9743e102b9dfa6cd75e94708d01640c9/Iris.csv -O data/raw/iris.csv

run_tests:
	pytest

docs_save:
	@echo Save documentation to docs...
	PYTHONPATH=scripts pdoc scripts -o docs

data/processed/xy.pkl: data/raw scripts/process.py
	@echo "Processing data..."
	python scripts/process.py

models/svc.pkl: data/processed/xy.pkl scripts/train_model.py
	@echo "Training model..."
	python scripts/train_model.py

notebooks/results.ipynb: models/svc.pkl scripts/run_notebook.py
	@echo "Running notebook..."
	python scripts/run_notebook.py

pipeline: data/processed/xy.pkl models/svc.pkl notebooks/results.ipynb

####################################################################

# Delete all compiled Python files and cache directories
clean_repo:
	@echo "Cleaning Repository..."
	del /S /Q .\*.pyc
	for /d /r %%d in (__pycache__) do @if exist "%%d" rd /S /Q "%%d"
	if exist .\.mypy_cache rmdir /S /Q .\.mypy_cache
	if exist .\.pytest_cache rmdir /S /Q .\.pytest_cache
	if exist .\.ruff_cache rmdir /S /Q .\.ruff_cache
	if exist .\catboost_info rmdir /S /Q .\catboost_info
	del .\.coverage
