import logging
import os
from datetime import datetime

# Define the directory name for log files
LOG_DIR = "logs"

# Create the full path for the log directory using os.getcwd()
LOG_DIR = os.path.join(os.getcwd(), LOG_DIR)

# Create the log directory if it doesn't exist
os.makedirs(LOG_DIR, exist_ok=True)

 # Get the current timestamp
CURRENT_TIME_STAMP = f"{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}"

# Define the log file name using the current timestamp
file_name = f"log_{CURRENT_TIME_STAMP}.log"

# Create the full path for the log file
log_file_path = os.path.join(LOG_DIR, file_name)

# Configure logging settings
logging.basicConfig(filename=log_file_path,  # Set the log file path
                    filemode="w",  # Set the file mode to write (overwrite if exists)
                    format='[%(asctime)s] %(name)s - %(levelname)s - %(message)s',  # Define the log message format
                    level=logging.INFO)  # Set the logging level to INFO

# Get the root logger
logger = logging.getLogger()