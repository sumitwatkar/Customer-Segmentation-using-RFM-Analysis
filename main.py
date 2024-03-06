import sys
from src.exception import CustomException
from src.logger import logging
from src.config.configuration import Configuration
from src.entity.artifact_entity import DataIngestionArtifact
from src.components.data_ingestion import DataIngestion
from src.pipeline.pipeline import Pipeline


# Define the Pipeline class
class Pipeline():
    
    # Constructor to initialize Pipeline object
    def __init__(self, config: Configuration = Configuration()) -> None:
        try:
            # Initialize Pipeline object with a Configuration object
            self.config = config
        except Exception as e:
            # If an exception occurs during initialization, raise a CustomException
            raise CustomException(e, sys) from e

    # Method to start the data ingestion process
    def start_data_ingestion(self) -> DataIngestionArtifact:
        
        try:
            # Create a DataIngestion object with data ingestion configuration from Pipeline's configuration
            data_ingestion = DataIngestion(data_ingestion_config=self.config.get_data_ingestion_config())
            
            # Initiate the data ingestion process and return the resulting artifact
            return data_ingestion.initiate_data_ingestion()
        
        except Exception as e:
            raise CustomException(e, sys) from e
    
    # Method to run the entire pipeline
    def run_pipeline(self):
        try:
            # Start the data ingestion process
            data_ingestion_artifact = self.start_data_ingestion()
        
        except Exception as e:
            raise CustomException(e, sys) from e

# Define the main function
def main():
    
    try:
       
        pipeline = Pipeline()  # Create a Pipeline object
        
        pipeline.run_pipeline() # Run the pipeline
    
    except Exception as e:
        logging.error(f"{e}")
        print(e)

# Entry point of the script
if __name__ == "__main__":
    main()