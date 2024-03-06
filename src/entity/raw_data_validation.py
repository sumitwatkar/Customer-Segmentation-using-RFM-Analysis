from src.exception import CustomException
from src.logger import logging
import sys
from src.utils.common import read_yaml_file
import pandas as pd
import collections
import yaml


# Class for data validation
class IngestedDataValidation:

    # Constructor method to initialize the class
    def __init__(self, validate_path, schema_path):
        
        try:
            # Initialize paths for validation and schema
            self.validate_path = validate_path
            self.schema_path = schema_path
            
            # Read schema data from YAML file
            self.data = read_yaml_file(self.schema_path)
        
        except Exception as e:
            raise CustomException(e, sys) from e

    
    # Method to validate filename
    def validate_filename(self, file_name) -> bool:
        
        try:
            # Check if filename matches schema's filename
            schema_file_name = self.data['FileName']
            if schema_file_name == file_name:
                return True

        except Exception as e:
            raise CustomException(e, sys) from e

    
    # Method to check for missing values in columns
    def missing_values_whole_column(self) -> bool:
        
        try:
            # Read data from CSV file
            df = pd.read_csv(self.validate_path)
            count = 0
            
            # Iterate through columns to check for missing values
            for columns in df:
                if (len(df[columns]) - df[columns].count()) == len(df[columns]):
                    count += 1
            
            # Return True if no missing values found, else False
            return True if (count == 0) else False

        except Exception as e:
            raise CustomException(e, sys) from e

    
    # Method to replace null values with 'NULL'
    def replace_null_values_with_nan(self) -> bool:
        
        try:
            # Read data from CSV file and replace null values with 'NULL'
            df = pd.read_csv(self.validate_path)
            df.fillna('NULL', inplace=True)
        
        except Exception as e:
            raise CustomException(e, sys) from e

    
    # Method to check if column names match schema
    def check_column_names(self) -> bool:
        
        try:
            # Read data from CSV file
            df = pd.read_csv(self.validate_path)
            df_column_names = df.columns
  
            # Get column names from schema
            schema_column_names = list(self.data['ColumnNames'].keys())
            
            # Check if column names match
            return True if (collections.Counter(df_column_names) == collections.Counter(schema_column_names)) else False

        except Exception as e:
            raise CustomException(e, sys) from e

    
    # Method to validate number of columns in dataframe
    def validate_number_of_columns(self, dataframe: pd.DataFrame) -> bool:
        try:
            # Get number of columns from schema
            number_of_columns = len(self._schema_config["columns"])
            logging.info(f"Required number of columns: {number_of_columns}")
            logging.info(f"Data frame has columns: {len(dataframe.columns)}")
            
            # Check if number of columns match
            if len(dataframe.columns) == number_of_columns:
                return True
            return False
        
        except Exception as e:
            raise CustomException(e, sys) from e

    
    # Method to validate data types of columns
    def validate_data_types(self, filepath, schema_path):
        flag = True  
        df = pd.read_csv(filepath)
        
        # Read schema data from YAML file
        with open(schema_path, "r") as file:
            schema_data = yaml.safe_load(file)

        # Get column names and expected data types from schema
        column_names = schema_data["ColumnNames"]
        
        print(column_names)

        # Iterate through columns to check data types
        for column, expected_type in column_names.items():
            if column not in df.columns:
                print(f"Column '{column}' not found in the dataset.")
            if not df[column].dtype == expected_type:
                flag = False
                print(f"Data type mismatch for column '{column}'. Expected {expected_type}, but found {df[column].dtype}.")

        return flag