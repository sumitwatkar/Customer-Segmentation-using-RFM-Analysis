from collections import namedtuple

# Defining DataIngestionArtifact namedtuple to represent artifacts generated during data ingestion
DataIngestionArtifact = namedtuple('DataIngestionArtifact',
                                   ['train_file_path','is_ingested','message'])

# Defining DataValidationArtifact namedtuple to represent artifacts generated during data validation
DataValidationArtifact = namedtuple("DataValidationArtifact",
["schema_file_path","is_validated","message","validated_train_path"])


# Defining DataTransformationArtifact namedtuple to represent artifacts generated during data transformation
DataTransformationArtifact = namedtuple("DataTransformationArtifact",["is_transformed",
                                                                    "message","feature_eng_train_file_path",
                                                                    "transformed_train_file_path",
                                                                    "preprocessed_object_file_path",
                                                                    "feature_engineering_object_file_path"])