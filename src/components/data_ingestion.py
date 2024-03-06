import os,sys
from six.moves import urllib  
from src.constants import *
from src.entity.config_entity import *
from src.entity.artifact_entity import *
from src.logger import logging
from src.exception import CustomException
import zipfile
import shutil


class DataIngestion:
    def __init__(self, data_ingestion_config: DataIngestionConfig):
        try:

            logging.info(f"{'>>'*5}Data Ingestion log started.{'<<'*5} \n\n")
            self.data_ingestion_config = data_ingestion_config  # Initializing data ingestion configuration
        
        except Exception as e:
            raise CustomException(e, sys) from e

    
    def download_data(self) -> str:
        try:
            # Retrieving raw data directory path from configuration
            raw_data_dir = self.data_ingestion_config.raw_data_dir

            # Logging raw data directory path
            logging.info(f" Raw Data directory : {raw_data_dir}")

            # Creating raw data directory if it doesn't exist
            os.makedirs(raw_data_dir, exist_ok=True)

            # Constructing download URL
            download_url = self.data_ingestion_config.dataset_download_url + "?raw=true"
            logging.info(f"Downloading the file from URL: {download_url}")

            # Downloading the zip file from the provided URL
            urllib.request.urlretrieve(download_url, os.path.join(raw_data_dir, "data.zip"))
            logging.info("File downloaded successfully.")

            # Extracting the downloaded zip file
            with zipfile.ZipFile(os.path.join(raw_data_dir, "data.zip"), "r") as zip_ref:
                zip_ref.extractall(raw_data_dir)
            logging.info("Zip file extracted successfully.")

            # Deleting the downloaded zip file
            os.remove(os.path.join(raw_data_dir, "data.zip"))

            # Initializing CSV file path variable
            csv_file_path = None

            # Getting the list of files in the raw data directory
            file_list = os.listdir(raw_data_dir)

            # Searching for the CSV file in the extracted files
            for file_name in file_list:
                if file_name.endswith(".csv"):
                    csv_file_path = os.path.join(raw_data_dir, file_name)
                    break

            # If CSV file found, update its path
            if csv_file_path is not None:
                csv_file_name = os.path.basename(csv_file_path)
                raw_file_path = os.path.join(raw_data_dir, csv_file_name)

            # Creating directory for ingested data
            ingest_file_path = os.path.join(self.data_ingestion_config.ingested_data_dir)
            os.makedirs(ingest_file_path, exist_ok=True)

            # Copying the extracted CSV file to ingested data directory
            shutil.copy2(raw_file_path, ingest_file_path)

            # Setting destination path for ingested data
            ingest_file_path = os.path.join(self.data_ingestion_config.ingested_data_dir, csv_file_name)

            # Logging successful data ingestion
            logging.info(f"File: {ingest_file_path} has been downloaded and extracted successfully.")

            # Creating data ingestion artifact with necessary information
            data_ingestion_artifact = DataIngestionArtifact(train_file_path=ingest_file_path,
                                                            is_ingested=True,
                                                            message=f"data ingestion completed successfully"
                                                            )
            logging.info(f"Data Ingestion Artifact:[{data_ingestion_artifact}]")

            return data_ingestion_artifact

        except Exception as e:
            raise CustomException(e, sys) from e

    def initiate_data_ingestion(self):
        try:
            # Initiating data download and ingestion process
            return self.download_data()

        except Exception as e:
            raise CustomException(e, sys) from e