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


# Defining DataTransformationArtifact namedtuple to represent artifacts generated during model trainer
ModelTrainerArtifact =namedtuple("ModelTrainerArtifact",[
                                                            "is_trained",
                                                            "message",
                                                            "model_selected",
                                                            "model_prediction_png",
                                                            "model_name",
                                                            "report_path",
                                                            "csv_file_path"
                                                        ])
# Defining ModelEvaluationArtifact namedtuple to represent artifacts generated during model evaluation
ModelEvaluationArtifact=namedtuple("ModelEvaluationArtifact",["model_name",
                                                              "Silhouette_score",
                                                              "selected_model_path",
                                                              "model_prediction_png",
                                                              "optimal_cluster",
                                                              "rfm_csv_path",
                                                              "model_report_path"])