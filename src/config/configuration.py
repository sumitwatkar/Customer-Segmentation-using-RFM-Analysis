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

    
    
    def get_data_validation_config(self) -> DataValidationConfig:
        try:
            # Get the directory where artifacts are stored from the training pipeline config
            artifact_dir = self.training_pipeline_config.artifact_dir

            # Construct the directory path for data validation artifacts based on the timestamp
            data_validation_artifact_dir=os.path.join(
                artifact_dir,
                DATA_VALIDATION_ARTIFACT_DIR,
                self.time_stamp
            )

            # Get the data validation configuration from the overall config information
            data_validation_config = self.config_info[DATA_VALIDATION_CONFIG_KEY]
            
            # Construct the path to the validated dataset
            validated_path=os.path.join(data_validation_artifact_dir,DATA_VALIDATION_VALID_DATASET)
            
            # Construct the path to the validated training data within the data validation artifact directory
            validated_train_path=os.path.join(data_validation_artifact_dir,validated_path,DATA_VALIDATION_TRAIN_FILE)
            
            # Construct the file path for the schema based on configuration information
            schema_file_path = os.path.join(
                ROOT_DIR,
                data_validation_config[DATA_VALIDATION_SCHEMA_DIR_KEY],
                data_validation_config[DATA_VALIDATION_SCHEMA_FILE_NAME_KEY]
            )
            
            # Create a DataValidationConfig object with the constructed paths
            data_validation_config = DataValidationConfig(
                schema_file_path=schema_file_path,validated_train_path=validated_train_path)
            
            return data_validation_config
        
        except Exception as e:
            raise CustomException(e,sys) from e

    
    def get_data_transformation_config(self) -> DataTransformationConfig:
        """
    Method to retrieve data transformation configuration.

    Returns:
        DataTransformationConfig: Object containing data transformation configuration.
    """
        try:
            # Define the directory paths for storing data transformation artifacts
            artifact_dir = self.training_pipeline_config.artifact_dir
            data_transformation_artifact_dir = os.path.join(artifact_dir, 
                                                            DATA_TRANSFORMATION_ARTIFACT_DIR, 
                                                            self.time_stamp)
            
            # Retrieve data transformation configuration from the  configuration
            data_transformation_config = self.config_info[DATA_TRANSFORMATION_CONFIG_KEY]

            # Define file paths for preprocessed and feature engineered data
            preprocessed_object_file_path = os.path.join(data_transformation_artifact_dir,
                                                        data_transformation_config[DATA_TRANSFORMATION_PREPROCESSING_DIR_KEY],
                                                        data_transformation_config[DATA_TRANSFORMATION_PREPROCESSING_FILE_NAME_KEY])

            feature_engineering_object_file_path = os.path.join(data_transformation_artifact_dir,
                                                        data_transformation_config[DATA_TRANSFORMATION_PREPROCESSING_DIR_KEY],
                                                        data_transformation_config[DATA_TRANSFORMATION_FEATURE_ENGINEERING_FILE_NAME_KEY])

            # Define directory path for transformed training data
            transformed_train_dir = os.path.join(data_transformation_artifact_dir,
                                                        data_transformation_config[DATA_TRANSFORMATION_DIR_NAME_KEY],
                                                        data_transformation_config[DATA_TRANSFORMATION_TRAIN_DIR_NAME_KEY])

            # Create DataTransformationConfig object with the obtained paths
            data_transformation_config = DataTransformationConfig(transformed_train_dir=transformed_train_dir,
                                                        preprocessed_object_file_path=preprocessed_object_file_path,
                                                        feature_engineering_object_file_path=feature_engineering_object_file_path)
            
            
            logging.info(f"Data Transformation Config: {data_transformation_config}")
            return data_transformation_config
        
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