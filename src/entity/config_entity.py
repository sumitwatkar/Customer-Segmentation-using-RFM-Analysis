from collections import namedtuple

# Defining DataIngestionConfig namedtuple to represent configuration parameters for data ingestion
DataIngestionConfig= namedtuple("DataIngestionConfig",
                                ["dataset_download_url",
                                 "ingested_data_dir",
                                 "raw_data_dir",
                                 ])

# Defining DataValidationConfig namedtuple to represent configuration parameters for data validation
DataValidationConfig = namedtuple("DataValidationConfig", ["schema_file_path","validated_train_path"])

# Defining TrainingPipelineConfig namedtuple to represent configuration parameters for the training pipeline
TrainingPipelineConfig= namedtuple("TrainingPipelineConfig",["artifact_dir"])