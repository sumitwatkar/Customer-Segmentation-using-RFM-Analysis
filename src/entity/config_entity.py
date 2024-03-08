from collections import namedtuple

# Defining DataIngestionConfig namedtuple to represent configuration parameters for data ingestion
DataIngestionConfig= namedtuple("DataIngestionConfig",
                                ["dataset_download_url",
                                 "ingested_data_dir",
                                 "raw_data_dir",
                                 ])

# Defining DataValidationConfig namedtuple to represent configuration parameters for data validation
DataValidationConfig = namedtuple("DataValidationConfig", ["schema_file_path","validated_train_path"])


# Defining DataTransformationConfig namedtuple to represent configuration parameters for data transformation
DataTransformationConfig = namedtuple("DataTransformationConfig",["transformed_train_dir",
                                                                  "preprocessed_object_file_path",
                                                                  "feature_engineering_object_file_path"])


# Defining ModelTrainerConfig namedtuple to represent configuration parameters for model trainer
ModelTrainerConfig = namedtuple("ModelTrainerConfig",["trained_model_directory",
                                                      "trained_model_file_path",
                                                      "png_location",
                                                      "model_config_path",
                                                      "report_path"])

# Defining SavedModelConfig namedtuple to represent configuration parameters for model evaluation
SavedModelConfig = namedtuple("SavedModelConfig",["saved_model_file_path",
                                                    "saved_model_csv",
                                                    "saved_report_file_path",
                                                    "saved_model_prediction_png"])

# Defining TrainingPipelineConfig namedtuple to represent configuration parameters for the training pipeline
TrainingPipelineConfig= namedtuple("TrainingPipelineConfig",["artifact_dir"])