from src.logger import logging
from src.components.data_ingestion import DataIngestion
from src.pipeline.pipeline import Pipeline

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