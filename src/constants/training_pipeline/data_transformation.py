import os

# Constants defining keys for accessing specific folder names and file paths.
PIKLE_FOLDER_NAME_KEY='prediction_files'
DATA_TRANSFORMATION_ARTIFACT_DIR = "data_transformation"

# Keys for accessing specific components within the data transformation configuration.
DATA_TRANSFORMATION_CONFIG_KEY = "data_transformation_config"
DATA_TRANSFORMATION_DIR_NAME_KEY = "transformed_dir"
DATA_TRANSFORMATION_TRAIN_DIR_NAME_KEY = "transformed_train_dir"
DATA_TRANSFORMATION_PREPROCESSING_DIR_KEY = "preprocessing_dir"
DATA_TRANSFORMATION_PREPROCESSING_FILE_NAME_KEY = "preprocessed_object_file_name"
DATA_TRANSFORMATION_FEATURE_ENGINEERING_FILE_NAME_KEY ="feature_engineering_object_file_name"

# Key for accessing the list of columns to be dropped during data transformation.
DROP_COLUMNS='drop_columns'

# Name of the transformation YAML file.
TRANFORMATION_YAML='transformation.yaml'

ROOT_DIR=os.getcwd()
CONFIG_DIR='config'

# File path for the transformation YAML file.
TRANSFORMATION_YAML_FILE_PATH=os.path.join(ROOT_DIR,CONFIG_DIR,TRANFORMATION_YAML)
