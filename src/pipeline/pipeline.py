import sys
from src.config.configuration import Configuration
from src.logger import logging
from src.exception import CustomException
from src.entity.artifact_entity import*
from src.components.data_ingestion import DataIngestion
from src.components.data_validation import DataValidation
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.components.model_evaluation import ModelEvaluation
from src.components.model_pusher import ModelPusher


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
        
    
    def start_data_transformation(self,data_ingestion_artifact: DataIngestionArtifact,
                                       data_validation_artifact: DataValidationArtifact) -> DataTransformationArtifact:
        """
        Starts the data transformation process.

        Returns:
        - DataTransformationArtifact: The artifact produced after data transformation.
        """
        try:
            data_transformation = DataTransformation(
                data_transformation_config = self.config.get_data_transformation_config(),
                data_validation_artifact = data_validation_artifact)

            return data_transformation.initiate_data_transformation()
        
        except Exception as e:
            raise CustomException(e,sys) from e


    def start_model_training(self,data_transformation_artifact: DataTransformationArtifact) -> ModelTrainerArtifact:
        """
        Starts the model training process.

        Returns:
        - ModelTrainerArtifact: The artifact produced after model training.
        """
        try:
            model_trainer = ModelTrainer(model_trainer_config=self.config.get_model_trainer_config(),
                                         
                                        data_transformation_artifact=data_transformation_artifact)   

            return model_trainer.initiate_model_training()
        
        except Exception as e:
            raise CustomException(e,sys) from e 

    
    def start_model_evaluation(self,data_validation_artifact:DataValidationArtifact,
                                 model_trainer_artifact:ModelTrainerArtifact):
        """
        Starts the model training process
        """
        try:
            model_eval = ModelEvaluation(data_validation_artifact,model_trainer_artifact)
                                         
            model_eval_artifact = model_eval.initiate_model_evaluation()
            return model_eval_artifact
        
        except  Exception as e:
            raise  CustomException(e,sys)


    def start_model_pusher(self,model_eval_artifact:ModelEvaluationArtifact):
                
                try:
                    model_pusher = ModelPusher(model_eval_artifact)
                    model_pusher_artifact = model_pusher.initiate_model_pusher()
                    return model_pusher_artifact
                
                except  Exception as e:
                    raise  CustomException(e,sys)
        

    def run_pipeline(self):
        """
        Runs the entire data pipeline.
        """
        try:
            # Start data ingestion process
            data_ingestion_artifact = self.start_data_ingestion()

            # Start data validation process
            data_validation_artifact=self.start_data_validation(data_ingestion_artifact=data_ingestion_artifact)
        
            # Start data transformation process
            data_transformation_artifact = self.start_data_transformation(data_ingestion_artifact=data_ingestion_artifact,
                                                             data_validation_artifact=data_validation_artifact)

            # Start model training process
            model_trainer_artifact = self.start_model_training(data_transformation_artifact=data_transformation_artifact)
        
            # Start model evaulation process
            model_eval_artifact = self.start_model_evaluation(data_validation_artifact, model_trainer_artifact)
        
            # Start model pusher process
            model_pusher_artifact = self.start_model_pusher(model_eval_artifact)
        
        except Exception as e:
            raise CustomException(e, sys) from e