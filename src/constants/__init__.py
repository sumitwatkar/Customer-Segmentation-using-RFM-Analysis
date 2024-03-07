
import os
from datetime import datetime

from src.constants.training_pipeline import *
from src.constants.training_pipeline.data_ingestion import *
from src.constants.training_pipeline.data_validation import *
from src.constants.training_pipeline.data_transformation import *

# Function to get the current timestamp in the format 'YYYY-MM-DD HH-MM-SS'
def get_current_time_stamp():
    return f"{datetime.now().strftime('%Y-%m-%d %H-%M-%S')}"

# Getting the current timestamp
CURRENT_TIME_STAMP = get_current_time_stamp()

# Getting the root directory path
ROOT_DIR = os.getcwd()

# Constants defining directory structure and file names
CONFIG_DIR= "config"
CONFIG_FILE_NAME= "config.yaml"

# Constructing the path to the configuration file
CONFIG_FILE_PATH= os.path.join(ROOT_DIR, CONFIG_DIR, CONFIG_FILE_NAME)