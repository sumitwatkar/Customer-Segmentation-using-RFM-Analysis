import sys, os
from src.config import *
from src.entity.config_entity import *
from src.entity.artifact_entity import *
from src.constants import *
from src.exception import CustomException
from src.logger import logging
from src.utils.common import read_yaml_file
from src.entity.raw_data_validation import IngestedDataValidation
import shutil



class DataValidation:
    def __init__(self,data_validation_config:DataValidationConfig,
                 data_ingestion_artifact:DataIngestionArtifact) -> None:
        try:
            # Initialize Data Validation process with logging
            logging.info(f"{'>>' * 30}Data Validation log started.{'<<' * 30} \n\n") 
            
            # Initialize instance variables
            self.data_validation_config = data_validation_config
            self.data_ingestion_artifact = data_ingestion_artifact
       
            # Set schema path from data validation config
            self.schema_path = self.data_validation_config.schema_file_path
            
            # Initialize IngestedDataValidation instance with train file path and schema path
            self.train_data = IngestedDataValidation(
                                validate_path=self.data_ingestion_artifact.train_file_path, schema_path=self.schema_path)
            
            # Set train file path
            self.train_path = self.data_ingestion_artifact.train_file_path
            
            # Set validated train file path from data validation config
            self.validated_train_path = self.data_validation_config.validated_train_path
            
        except Exception as e:
            raise CustomException(e,sys) from e


    def isFolderPathAvailable(self) -> bool:
        try:
            # Check if train folder path exists
            isfolder_available = False
            train_path = self.train_path
            
            if os.path.exists(train_path):
                    isfolder_available = True
            return isfolder_available
        
        except Exception as e:
            raise CustomException(e, sys) from e     


    def is_Validation_successfull(self):
        try:
            # Perform data validation process
            validation_status = True
            logging.info("Validation Process Started")
            
            # Check if folder path is available
            if self.isFolderPathAvailable() == True:
                
                # Get train file name
                train_filename = os.path.basename(
                    self.data_ingestion_artifact.train_file_path)

                # Validate train file name
                is_train_filename_validated = self.train_data.validate_filename(
                    file_name=train_filename)

                # Check if train column names are valid
                is_train_column_name_same = self.train_data.check_column_names()
                
                # Validate data types of train data
                validating_train_data_types=self.train_data.validate_data_types(filepath=self.train_path,
                                                                                schema_path=self.data_validation_config.schema_file_path)

                # Check for missing values in whole column of train data
                is_train_missing_values_whole_column = self.train_data.missing_values_whole_column()
                
                # Replace null values with NaN in train data
                self.train_data.replace_null_values_with_nan()
                
                logging.info(
                    f"Train_set status: "
                    f"is Train filename validated? {is_train_filename_validated} | "
                    f"is train column name validated? {is_train_column_name_same} | "
                    f"whole missing columns? {is_train_missing_values_whole_column}"
                    f"Data type validation? {validating_train_data_types}"
                )

                # If all validation checks are successful
                if is_train_filename_validated  & is_train_column_name_same & is_train_missing_values_whole_column & validating_train_data_types :
                    
                    # Create directory for validated train data
                    os.makedirs(self.validated_train_path, exist_ok=True)
                    
                    # Get file name from schema data
                    schema_data=read_yaml_file(self.schema_path)
                    file_name=schema_data['FileName']

                    # Copy validated train data to validated train path
                    shutil.copy(self.train_path, self.validated_train_path)
                    self.validated_train_path=os.path.join(self.validated_train_path,file_name)
               
                    # Log successful validation and export of validated train dataset
                    logging.info(f"Exported validated train dataset to file: [{self.validated_train_path}]")
                    return validation_status,self.validated_train_path
                
                else:
                    # If validation checks fail, set validation status to False and raise ValueError
                    validation_status = False
                    logging.info("Check your training data! Validation Failed")
                    raise ValueError(
                        "Check your training data! Validation failed")
                
            return validation_status,"NONE","NONE"
        
        except Exception as e:
            raise CustomException(e, sys) from e      
        

    def initiate_data_validation(self):
        try:
            # Initiate data validation process
            
            # Check if validation is successful and get validated train path
            is_validated, validated_train_path = self.is_Validation_successfull()
            
            # Create DataValidationArtifact instance with validation results
            data_validation_artifact = DataValidationArtifact(
                schema_file_path=self.schema_path,
                is_validated=is_validated,
                message="Data_validation_performed ",
                validated_train_path=validated_train_path
            )
            # Log data validation artifact
            logging.info(f"Data validation artifact: {data_validation_artifact}")
            return data_validation_artifact

        except Exception as e:
            raise CustomException(e, sys) from e


    def __del__(self):
        logging.info(f"{'>>' * 5}Data Validation log completed.{'<<' * 5}")