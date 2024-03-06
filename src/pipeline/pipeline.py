import sys
from src.config.configuration import Configuration
from src.logger import logging
from src.exception import CustomException
from src.entity.artifact_entity import*
from src.components.data_ingestion import DataIngestion
from src.components.data_validation import DataValidation


class Pipeline():
    """
    A class representing a data pipeline.

    Attributes:
    - config: Configuration object containing pipeline configuration settings.
    """

    def __init__(self, config: Configuration = Configuration()) -> None:
        """
        Initializes a Pipeline object.

        Args:
        - config: Configuration object containing pipeline configuration settings. Default is an instance of Configuration.
        """
        try:
            self.config = config  # Setting the configuration attribute.
        except Exception as e:
            raise CustomException(e, sys) from e


    def start_data_ingestion(self) -> DataIngestionArtifact:
        """
        Starts the data ingestion process.

        Returns:
        - DataIngestionArtifact: The artifact produced after data ingestion.
        """
        try:
            data_ingestion = DataIngestion(data_ingestion_config=self.config.get_data_ingestion_config())  # Initializing DataIngestion object with data ingestion configuration
            return data_ingestion.initiate_data_ingestion()
        
        except Exception as e:
            raise CustomException(e, sys) from e

    
    def start_data_validation(self, data_ingestion_artifact:DataIngestionArtifact)-> DataValidationArtifact:
        """
        Starts the data validation process.

        Returns:
        - DataValidationArtifact: The artifact produced after data validation.
        """
        try:
            # Initializing DataValidation object with data validation configuration
            data_validation = DataValidation(data_validation_config=self.config.get_data_validation_config(),
                                             data_ingestion_artifact=data_ingestion_artifact)
            return data_validation.initiate_data_validation()
        
        except Exception as e:
            
            raise CustomException(e, sys) from e
        
    
    def run_pipeline(self):
        """
        Runs the entire data pipeline.
        """
        try:
            # Start data ingestion process
            data_ingestion_artifact = self.start_data_ingestion()

            # Start data validation process
            data_validation_artifact=self.start_data_validation(data_ingestion_artifact=data_ingestion_artifact)
        
        except Exception as e:
            raise CustomException(e, sys) from e