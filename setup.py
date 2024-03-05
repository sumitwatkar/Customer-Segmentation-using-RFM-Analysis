from setuptools import setup, find_packages
from typing import List

# Project metadata
PROJECT_NAME = "Retail Customer Segmentation"
VERSION = "0.0.1"
DESCRIPTION = "Understanding customer behaviour using RFM analysis"
AUTHOR_NAME = "Sumit Watkar"
AUTHOR_EMAIL = "watkar.sumit@gmail.com"

# Name of the requirements file
REQUIREMENTS_FILE_NAME = "requirements.txt"

# String to be removed from requirements if present
HYPHEN_E_DOT = "-e ."

# Function to read requirements from requirements file and return as a list
def get_requirements_list() -> List[str]:
    with open(REQUIREMENTS_FILE_NAME) as requirement_file:
        requirement_list = requirement_file.readlines()
        requirement_list = [requirement_name.replace("\n", "") for requirement_name in requirement_list]

        # Remove the "-e ." if present
        if HYPHEN_E_DOT in requirement_list:
            requirement_list.remove(HYPHEN_E_DOT)

        return requirement_list

# Setup function to define package metadata and dependencies
setup(
    name=PROJECT_NAME,
    version=VERSION,
    description=DESCRIPTION,
    author=AUTHOR_NAME,
    author_email=AUTHOR_EMAIL,
    packages=find_packages(),  # Automatically find packages under the project directory
    install_requires=get_requirements_list()  # Define required dependencies from requirements file
)