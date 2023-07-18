"""Create a virtual environment by running the file from the terminal.

# ! IMPORTANT: when this file is first opened, select the appropriate Python
# ! environment as requested by, for example, VS Code.

This script first creates a new virtual environment called "venv". It then
activates the virtual environment, updates pip, and installs the packages
specified in the packages list.

Next, it creates the folders specified in the folders list and adds a
'.gitkeep' file to each one to ensure they are tracked in git.

Finally, it creates a scripts folder and adds '__init__.py' and 'main.py' files
to it. Note that the 'os.makedirs()' function is used to create the folder if
it doesn't already exist, and the 'exist_ok=True' argument ensures that the
function doesn't raise an exception if the folder already exists.

NOTE: this script assumes that you are running it from the root directory
where you want to create the virtual environment and the other files/folders.
You can modify the script to change the names of the virtual environment and
the folders as needed.

BUG Alert !!! BUG
The creation of the virtual environment cannot be run automatically through
this script. I tried many options suggested by ChatGTP and other StackOverflow
threads. Follow the instructions below for more properly activating the 'venv'.

BUG The same applies to the automatic update of the 'pip' package. BUG
"""

import os
import subprocess


# Create a  virtual environment
subprocess.run(args=["python", "-m", "venv", "venv"])
print("\nVirtual environment created.\n")


# Activate virtual environment
# ! BUG: command below NOT working !
# * It is okay, do the following:
# *   - copy/paste the command '.\venv\Scripts\Activate.ps1' in the CLI
# *   - run it
# *   - comment out the command in the script
# *   - rerun the script
# subprocess.run(args=[".\\venv\\Scripts\\Activate.ps1"])
# print("\nVirtual environment activated.\n")
# * Copy/paste the following code in the CLI
# * .\venv\Scripts\Activate.ps1


# Upgrade pip
subprocess.run(args=["python", "-m", "pip", "install", "--upgrade", "pip"])
print("\nPip package updated.\n")
# ! BUG: command above NOT working - must be done manually...
# * Copy/paste the following code in the CLI
# * python -m pip install --upgrade pip

# Install required packages
packages = [
    "black", "refurb", "sourcery", "isort", "pycodestyle", "pydocstyle",
    "pipreqs", "pipdeptree", "python-dotenv"
]
subprocess.run(args=["pip", "install"] + packages)
print("\nPackages installed.\n")


# Create 'requirements.txt' file using the 'pipreqs' library
subprocess.run(args=["pipreqs", ".", "--force"])
print("\n'requirexements.txt' file created.\n")


# Create required folders and .gitkeep files
folders = ["scripts", "figures", "output", "sources", "tests", "prod"]
for folder in folders:
    os.makedirs(folder, exist_ok=True)
    with open(file=os.path.join(folder, ".gitkeep"), mode="w") as f:
        pass
print("\nFolders and '.gitkeep' files created.\n")


# Create __init__.py and main.py files in the 'scripts' folder
scripts = ["__init__.py", "main.py"]
for script in scripts:
    with open(file=os.path.join("scripts", script), mode="w") as f:
        pass
print("\n'__init__.py' and 'main.py' files created inside 'scripts' folder.\n")


# Display the list of installed packages
print("\nList of installed packages:\n")
subprocess.run(args=["pip", "list"])
