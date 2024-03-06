import os,sys
from src.constants import *
from src.entity.config_entity import *
from src.entity.artifact_entity import *
from src.logger import logging
from src.exception import CustomException
from src.utils.common import read_yaml_file


class Configuration:
# Constructor method to initialize the Configuration object
    def __init__(self,
                 config_file_path:str = CONFIG_FILE_PATH,
                 current_time_stamp:str =CURRENT_TIME_STAMP) -> None:
        try:
            self.config_info = read_yaml_file(file_path = config_file_path) # Reading the YAML configuration file and storing its contents
            self.training_pipeline_config = self.get_training_pipeline_config() # Retrieving the training pipeline configuration
            self.time_stamp = current_time_stamp # Storing the current timestamp

        except Exception as e:
            raise CustomException(e,sys)from e
        
    # Method to retrieve data ingestion configuration
    def get_data_ingestion_config(self) -> DataIngestionConfig:
        
        try:

            artifact_dir = self.training_pipeline_config.artifact_dir # Retrieving artifact directory from training pipeline configuration
            
            # Constructing directory path for data ingestion artifact
            data_ingestion_artifact_dir=os.path.join(
                artifact_dir,
                DATA_INGESTION_ARTIFACT_DIR,
                self.time_stamp
            )

            data_ingestion_info = self.config_info[DATA_INGESTION_CONFIG_KEY] # Retrieving data ingestion information from the configuration

            dataset_download_url = data_ingestion_info[DATA_INGESTION_DOWNLOAD_URL_KEY] # Retrieving dataset download URL from data ingestion information

            # Constructing directory path for raw data
            raw_data_dir = os.path.join(data_ingestion_artifact_dir,
                                        data_ingestion_info[DATA_INGESTION_RAW_DATA_DIR_KEY]
                                        )
            
            # Constructing directory path for ingested data
            ingested_data_dir = os.path.join(
                data_ingestion_artifact_dir,
                data_ingestion_info[DATA_INGESTION_INGESTED_DIR_NAME_KEY]
            )

            # Creating DataIngestionConfig object with retrieved information
            data_ingestion_config=DataIngestionConfig(
                dataset_download_url=dataset_download_url,
                raw_data_dir=raw_data_dir, 
                ingested_data_dir=ingested_data_dir     
            )
            logging.info(f"Data Ingestion config: {data_ingestion_config}")
            return data_ingestion_config
        
        except Exception as e:
            raise CustomException(e,sys) from e


    # Method to retrieve training pipeline configuration
    def get_training_pipeline_config(self)->TrainingPipelineConfig:
        
        try:
            training_pipeline_config = self.config_info[TRAINING_PIPELINE_CONFIG_KEY] # Retrieving training pipeline configuration from the configuration

            # Constructing artifact directory path
            artifact_dir = os.path.join(ROOT_DIR,
                                        training_pipeline_config[TRAINING_PIPLELINE_NAME_KEY],
                                        training_pipeline_config[TRAINING_PIPELINE_ARTIFACT_DIR_KEY])  
    
            # Creating TrainingPipelineConfig object with the artifact directory
            training_pipeline_config = TrainingPipelineConfig(artifact_dir=artifact_dir)

            logging.info(f"Training Pipeline Configuration completed : {training_pipeline_config}")

            return training_pipeline_config

        except Exception as e:
            raise CustomException(e,sys) from e