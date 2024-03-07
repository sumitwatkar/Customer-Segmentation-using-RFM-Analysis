import sys
import yaml
from src.constants import *
from src.exception import CustomException
import pandas as pd
import dill


def read_yaml_file(file_path:str)->dict:
    """
    Reads a YAML file and returns the contents as a dictionary.
    file_path: str
    """
    try:
        with open(file_path, 'rb') as yaml_file:
            return yaml.safe_load(yaml_file)
    except Exception as e:
        raise CustomException(e,sys) from e
    

def save_data(file_path:str, data:pd.DataFrame):
    """
    Function to save pandas DataFrame to a CSV file.

    Parameters:
    - file_path (str): The path where the CSV file will be saved.
    - data (pd.DataFrame): The DataFrame to be saved.
    """
    try:
        # Extract the directory path from the file_path
        dir_path = os.path.dirname(file_path)
        
        # Create directory if it doesn't exist
        os.makedirs(dir_path, exist_ok=True)
        
        # Save DataFrame to CSV file without index
        data.to_csv(file_path, index=None)
    
    except Exception as e:
        raise CustomException(e, sys) from e


def save_object(file_path: str, obj):
    """
    Function to save an object to a binary file using dill serialization.

    Parameters:
    - file_path (str): The path where the object file will be saved.
    - obj: The object to be saved.
    """
    try:
        # Extract the directory path from the file_path
        dir_path = os.path.dirname(file_path)
        
        # Create directory if it doesn't exist
        os.makedirs(dir_path, exist_ok=True)
        
        # Serialize and save the object to the file using dill
        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)
    
    except Exception as e:
        raise CustomException(e, sys) from e
    

def write_yaml_file(file_path: str, data: dict = None):
    try:
        # Create the directory structure for the file path if it does not exist
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        # Open the file in write mode
        with open(file_path, "w") as yaml_file:
            # If data is provided, write it to the YAML file using YAML.dump()
            if data is not None:
                yaml.dump(data, yaml_file)
    
    except Exception as e:
        raise CustomException(e, sys)