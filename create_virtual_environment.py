"""Create a virtual environment by running the file from the terminal.

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
"""

import os
import subprocess

# # Create a  virtual environment
# subprocess.run(["python", "-m", "venv", "venv"])
# print("Virtual environment created.")

# Install the expected Python version
subprocess.run(["python", "-m", "pip", "install", "python==3.7.0"])

# Activate virtual environment
# subprocess.run(["source", "./venv/Scripts/Activate.ps1"])
subprocess.run(["./venv/Scripts/Activate.ps1"])
# # Alternative code
# activate_path = os.path.join("venv", "Scripts", "activate.ps1")
# subprocess.run([activate_path])
print("Virtual environment activated.")

# Upgrade pip
subprocess.run(["python", "-m", "pip", "install", "--upgrade", "pip"])
print("Pip package updated.")

# Install required packages
packages = ["black", "mypy"]
subprocess.run(["pip", "install"] + packages)
print("Packages installed.")

# Create required folders and .gitkeep files
folders = ["scripts", "images", "output", "sources"]
for folder in folders:
    os.makedirs(folder, exist_ok=True)
    with open(os.path.join(folder, ".gitkeep"), "w") as f:
        pass
print("Folders and .gitkeep files created.")

# Create __init__.py and main.py files in the scripts folder
script_folder = "scripts"
os.makedirs(script_folder, exist_ok=True)
scripts = ["__init__.py", "main.py"]
for script in scripts:
    with open(os.path.join(script_folder, script), "w") as f:
        pass
# # Alternative code if for-loop above not working
# with open(os.path.join(script_folder, "__init__.py"), "w") as f:
#     pass
# with open(os.path.join(script_folder, "main.py"), "w") as f:
#     pass
print("Script folder created with __init__.py and main.py files.")
