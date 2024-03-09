import os
import yaml
import pandas as pd


# Function definition to write schema information into a YAML file based on a CSV file
def write_schema_yaml(csv_file):

    # Reading the CSV file into a pandas DataFrame
    df = pd.read_csv(csv_file)
    
    # Getting the number of columns in the DataFrame
    num_cols = len(df.columns)
    
    # Converting column names and data types to lists for YAML compatibility
    column_names = df.columns.tolist()  # Extracting column names
    column_dtypes = df.dtypes.astype(str).tolist()  # Extracting column data types and converting them to strings

    # Creating a schema dictionary to store information about the CSV file
    schema = {
        "FileName": os.path.basename(csv_file),  # Storing the base name of the CSV file
        "NumberOfColumns": num_cols,  # Storing the number of columns in the DataFrame
        "ColumnNames": dict(zip(column_names, column_dtypes))  # Storing column names and their corresponding data types as a dictionary
    }

    # Writing the schema dictionary into a YAML file named 'schema.yaml'
    with open("schema.yaml", "w") as file:
        yaml.dump(schema, file)  # Dumping the schema dictionary into the YAML file

# Calling the function with a CSV file path as an argument to generate the schema YAML file
write_schema_yaml(r"data\\retail.csv")