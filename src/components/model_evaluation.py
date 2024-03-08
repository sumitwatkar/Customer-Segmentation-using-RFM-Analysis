import sys, os
from src.exception import CustomException
from src.logger import logging
from src.entity.config_entity import *
from src.entity.artifact_entity import *
from src.constants import *
from src.constants.training_pipeline import *
from src.config.configuration import Configuration
from src.utils.common import read_yaml_file,load_object


class ModelEvaluation:
    """Class to handle model evaluation"""

    def __init__(self, data_validation_artifact: DataValidationArtifact, model_trainer_artifact: ModelTrainerArtifact):
        """Constructor method to initialize model evaluation with required artifacts"""
        try:
            # Initializing data validation and model trainer artifacts
            self.data_validation_artifact = data_validation_artifact
            self.model_trainer_artifact = model_trainer_artifact
            
            # Initializing configuration object
            self.config = Configuration()
            
            # Loading saved model configuration
            self.saved_model_config = self.config.saved_model_config()
            
        except Exception as e:
            raise CustomException(e, sys)
        
    def initiate_model_evaluation(self) -> ModelEvaluationArtifact:
        """Method to initiate model evaluation process"""
        try:
            logging.info(" Model Evaluation Started ")
 
            # Retrieving paths and data from model trainer artifact
            model_trained_artifact_path = self.model_trainer_artifact.model_selected
            model_trained_report = self.model_trainer_artifact.report_path
            artifact_model_prediction_png = self.model_trainer_artifact.model_prediction_png
            artifact_rfm_table = self.model_trainer_artifact.csv_file_path

            # Retrieving paths from saved model configuration
            saved_model_path = self.saved_model_config.saved_model_file_path
            saved_model_report_path = self.saved_model_config.saved_report_file_path
            saved_model_prediction_png = self.saved_model_config.saved_model_prediction_png
            saved_model_rfm_table = self.saved_model_config.saved_model_csv
                        
            logging.info(f" Artifact Trained model : {model_trained_artifact_path}")
            
            # Creating saved_models directory if not exists
            logging.info("Saved_models directory .....")
            os.makedirs(SAVED_MODEL_DIRECTORY, exist_ok=True)
            
            if not os.listdir(SAVED_MODEL_DIRECTORY):
                # If saved_models directory is empty, select the trained model
                model_trained_report_data = read_yaml_file(file_path=model_trained_report)
                artifact_model_Silhouette_score = float(model_trained_report_data['Silhouette_score'])
                model_name = model_trained_report_data['Model_name']
                Silhouette_score = artifact_model_Silhouette_score
                cluster = int(model_trained_report_data['number_of_clusters'])
                model_path = model_trained_artifact_path
                model_report_path = model_trained_report
                png_path = artifact_model_prediction_png
                rfm_csv_path = artifact_rfm_table
                
            else:
                # If saved_models directory is not empty, compare performance of trained model with saved model
                saved_model_report_data = read_yaml_file(file_path=saved_model_report_path)
                model_trained_report_data = read_yaml_file(file_path=model_trained_report)
                saved_model = load_object(file_path=saved_model_path)
                artifact_model = load_object(file_path=model_trained_artifact_path)
                saved_model_Silhouette_score = float(saved_model_report_data['Silhouette_score'])
                artifact_model_Silhouette_score = float(model_trained_report_data['Silhouette_score'])
                
                if artifact_model_Silhouette_score > saved_model_Silhouette_score:
                    # If trained model performs better, select it
                    logging.info("Trained model outperforms the saved model!")
                    model_path = model_trained_artifact_path
                    model_report_path = model_trained_report
                    png_path = artifact_model_prediction_png
                    model_name = model_trained_report_data['Model_name']
                    Silhouette_score = float(model_trained_report_data['Silhouette_score'])
                    cluster = int(model_trained_report_data['number_of_clusters'])
                    rfm_csv_path = artifact_rfm_table
                    logging.info(f"Model Selected : {model_name}")
                    logging.info(f"Silhouette_score : {Silhouette_score}")
                  
                elif artifact_model_Silhouette_score < saved_model_Silhouette_score:
                    # If saved model performs better, select it
                    logging.info("Saved model outperforms the trained model!")
                    model_path = saved_model_path
                    model_report_path = saved_model_report_path
                    png_path = saved_model_prediction_png
                    model_name = saved_model_report_data['Model_name']
                    Silhouette_score = float(saved_model_report_data['Silhouette_score'])
                    cluster = int(saved_model_report_data['number_of_clusters'])
                    rfm_csv_path = saved_model_rfm_table
                    logging.info(f"Model Selected : {model_name}")
                    logging.info(f"Silhouette Score : {Silhouette_score}")
            
                else:
                    # If both models perform equally, select saved model
                    logging.info("Both models have the same Silhouette_score.")
                    model_path = saved_model_path
                    model_report_path = saved_model_report_path
                    png_path = saved_model_prediction_png
                    cluster = int(saved_model_report_data['number_of_clusters'])
                    model_name = saved_model_report_data['Model_name']
                    rfm_csv_path = saved_model_rfm_table
                    Silhouette_score = float(saved_model_report_data['Silhouette_score'])
                    logging.info(f"Model Selected : {model_name}")
                    logging.info(f"Silhouette Score : {Silhouette_score}")
                
            # Creating ModelEvaluationArtifact object
            model_evaluation = ModelEvaluationArtifact(model_name=model_name, 
                                                       Silhouette_score=Silhouette_score,
                                                       selected_model_path=model_path, 
                                                       model_report_path=model_report_path,
                                                       optimal_cluster=cluster,
                                                       model_prediction_png=png_path,
                                                       rfm_csv_path=rfm_csv_path)
            
            logging.info("Model evaluation completed successfully!")
            return model_evaluation
        except Exception as e:
        
            logging.error("Error occurred during model evaluation!")
            raise CustomException(e, sys) from e

    def __del__(self):
        """Destructor method to log completion of model evaluation"""
        logging.info(f"{'*'*5} Model evaluation log completed {'*'*5}\n\n")