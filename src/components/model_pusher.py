import sys, os
import yaml
import shutil
from src.logger import logging
from src.exception import CustomException
from src.entity.artifact_entity import ModelEvaluationArtifact,ModelPusherArtifact
from src.utils.common import load_object,save_object
from src.constants.training_pipeline import *
from src.constants import *

               
          
class ModelPusher:

    def __init__(self,model_eval_artifact:ModelEvaluationArtifact):

        try:
            self.model_eval_artifact = model_eval_artifact
        except  Exception as e:
            raise CustomException(e, sys)
      
            
            
            
    def initiate_model_pusher(self):
        try:
     
            model_path = self.model_eval_artifact.selected_model_path
            logging.info(f" Model path : {model_path}")
            model = load_object(file_path=model_path)
            file_path=os.path.join(ROOT_DIR,SAVED_MODEL_DIRECTORY,'model.pkl')
            
            save_object(file_path=file_path, obj=model)
            logging.info("Model saved.")
            

            model_name = self.model_eval_artifact.model_name
            Silhouette_score = self.model_eval_artifact.Silhouette_score
            optimal_cluster=self.model_eval_artifact.optimal_cluster
           

            report = {'Model_name': model_name, 'Silhouette_score': Silhouette_score,'number_of_clusters':optimal_cluster}

            logging.info(str(report))
            
      
            file_path=os.path.join(ROOT_DIR,SAVED_MODEL_DIRECTORY,MODEL_REPORT_FILE)
            logging.info(f"Report Location: {file_path}")

           
            with open(file_path, 'w') as file:
                yaml.dump(report, file)

            logging.info("Report saved as YAML file.")
            
         
            image_file_path=self.model_eval_artifact.model_prediction_png
            file_path=os.path.join(ROOT_DIR,SAVED_MODEL_DIRECTORY)
            try:                
                shutil.copy2(image_file_path, file_path)
            except Exception as e:
             
                print("An error occurred:", e)
                
         
            rfm_csv_path=self.model_eval_artifact.rfm_csv_path
            file_path=os.path.join(ROOT_DIR,SAVED_MODEL_DIRECTORY,'rfm.csv')
            try:
                shutil.copy2(rfm_csv_path, file_path)
                print("File copied successfully!")
            except Exception as e:
                # Handle the exception
                print("An error occurred:", e)
                
                
 
                    

            model_pusher_artifact = ModelPusherArtifact(message="Model Pushed succeessfully")
            return model_pusher_artifact
        except  Exception as e:
            raise CustomException(e, sys)
    
        
    def __del__(self):
        logging.info(f"{'*'*5} Model Pusher log completed {'*'*5}\n\n")