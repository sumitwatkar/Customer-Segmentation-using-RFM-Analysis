import sys, os
import pandas as pd
import numpy as np
from src.logger import logging
from src.exception import CustomException
from src.entity.artifact_entity import *
from src.entity.config_entity import *
from src.utils.common import read_yaml_file, save_data, save_object
from src.constants import *
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler


# Custom Transformer class for feature engineering
class Feature_Engineering(BaseEstimator, TransformerMixin):
    
    # Constructor
    def __init__(self, drop_columns):
        
        logging.info(f" {'>>'*5} Feature Engineering Started {'<<'*5} \n\n")
        self.columns_to_drop = drop_columns
        
        
        logging.info(f" Numerical Columns , Categorical Columns , Target Column initialised in Feature engineering Pipeline ")
            
    # Method to drop specified columns
    def drop_columns(self, X: pd.DataFrame):
        try:
            columns = X.columns
            logging.info(f"Columns before drop  {columns}")
            drop_column_labels = self.columns_to_drop
            
            logging.info(f" Dropping Columns {drop_column_labels} ")
            X = X.drop(columns=drop_column_labels, axis=1)  # Dropping specified columns
            return X
        
        except Exception as e:
            raise CustomException(e, sys) from e
    
    # Method to drop rows with NaN values
    def drop_rows_with_nan(self, X: pd.DataFrame):
        logging.info(f"Shape before dropping NaN values: {X.shape}")
        X = X.dropna()  # Dropping rows with NaN values
        logging.info(f"Shape after dropping NaN values: {X.shape}")
        logging.info("Dropped NaN values.")
        return X
    
    # Method to drop duplicate rows
    def drop_duplicates(self, X: pd.DataFrame):
        print(" Drop duplicate value")
        X = X.drop_duplicates()  # Dropping duplicate rows
        return X
    
    # Method to separate date and time from a column
    def separate_date_time(self, X, column_label):
        X[column_label] = pd.to_datetime(X[column_label])
        X['Invoice_Date'] = X[column_label].dt.date
        X['Invoice_Date'] = pd.to_datetime(X['Invoice_Date'])
        X.drop(column_label, axis=1, inplace=True)
        return X
    
    # Method to remove duplicate rows keeping the last occurrence
    def remove_duplicate_rows_keep_last(self, X):
        logging.info(f"DataFrame shape before removing duplicates: {X.shape}")
        num_before = len(X)
        X.drop_duplicates(inplace=True)  # Dropping duplicate rows
        num_after = len(X)
        num_duplicates = num_before - num_after
        logging.info(f"Removed {num_duplicates} duplicate rows")
        logging.info(f"DataFrame shape after removing duplicates: {X.shape}")
        return X
    
    # Method to convert specific string values to NaN
    def convert_nan_null_to_nan(self, X: pd.DataFrame):
        X.replace(["NAN", "NULL", "nan"], np.nan, inplace=True)  # Converting specific string values to NaN
        return X
    
    # Method to calculate total price
    def calculate_total_price(self, X):
        try:
            X['TotalPrice'] = X['Quantity'] * X['UnitPrice']  # Calculating total price
        except KeyError:
            logging.error("One or more required columns (Quantity, UnitPrice) not found in the DataFrame.")
        except Exception as e:
            logging.error("An error occurred while calculating the Total Price: {}".format(str(e)))
        return X
    
    # Method to drop rows with negative values in a specified column
    def drop_negative_rows(self, X, column_name):
        return X[X[column_name] >= 0]  # Dropping rows with negative values
    
    # Method to get maximum and minimum dates from a specified column
    def get_max_min_dates(self, X: pd.DataFrame, date_column):
        max_date = X[date_column].max()  # Getting maximum date
        min_date = X[date_column].min()  # Getting minimum date
        return X, max_date, min_date
    
    # Method to add a recency column based on specified date column
    def add_recency_column(self, X, date_column, min_date, max_date):
        X['recency'] = (max_date - X[date_column]).dt.days  # Adding recency column
        recency_table = X.groupby('CustomerID')['recency'].min().reset_index()  # Creating recency table
        logging.info(" Recency Table Created")
        logging.info(f"Shape of recency_table : {recency_table.shape}")
        return recency_table
    
    # Method to add a frequency column
    def add_frequency_column(self, X):
        frequency_table = X.groupby('CustomerID').count()['InvoiceNo'].to_frame().reset_index()  # Creating frequency table
        frequency_table.rename(columns={'InvoiceNo': 'frequency'}, inplace=True)
        logging.info(" Frequency Table Created")
        logging.info(f"Shape of frequency_table : {frequency_table.shape}")
        return frequency_table
    
    # Method to add a monetary column
    def add_monetory_column(self, X):
        monetary_table = X.groupby('CustomerID')['TotalPrice'].sum().rename('monetary').reset_index()  # Creating monetary table
        monetary_table.rename(columns={'TotalPrice': 'monetary'}, inplace=True)
        logging.info(" Monetory Table Created")
        logging.info(f"Shape of monetary_table : {monetary_table.shape}")
        return monetary_table
    
    # Method to merge tables into an RFM table
    def merge_tables(self, X, recency_table, frequency_table, monetary_table):
        logging.info("Merging to form RMF Table ...")
        data = X.groupby('CustomerID').first().reset_index()
        customer_ids = data['CustomerID']
        rfm_table = pd.merge(customer_ids, recency_table, on='CustomerID')  # Merging tables
        rfm_table = pd.merge(rfm_table, frequency_table, on='CustomerID')
        rfm_table = pd.merge(rfm_table, monetary_table, on='CustomerID')
        logging.info(f"Tables merged : Columns - {rfm_table.columns}")
        return rfm_table
    
    # Method to run data modification steps
    def run_data_modification(self, data):
        X = data.copy()
        X = self.remove_duplicate_rows_keep_last(X)  # Remove duplicate rows
        X = self.drop_columns(X=data)  # Drop specified columns
        X = self.convert_nan_null_to_nan(X)  # Convert specific string values to NaN
        X = self.drop_rows_with_nan(X)  # Drop rows with NaN values
        X = self.drop_negative_rows(X, 'Quantity')  # Drop rows with negative values
        X = X[X['UnitPrice'] >= 0]  # Drop rows with negative Unit Price
        X = self.drop_rows_with_nan(X)  # Drop rows with NaN values
        X = self.calculate_total_price(X=X)  # Calculate total price
        X = self.separate_date_time(X, 'InvoiceDate')  # Separate date and time
        X, max_date, min_date = self.get_max_min_dates(X, 'Invoice_Date')  # Get maximum and minimum dates
        recency_table = self.add_recency_column(X, date_column='Invoice_Date', min_date=min_date, max_date=max_date)  # Add recency column
        frequency_table = self.add_frequency_column(X=X)  # Add frequency column
        monetory_table = self.add_monetory_column(X=X)  # Add monetary column
        rmf_table = self.merge_tables(X=X, recency_table=recency_table, frequency_table=frequency_table, monetary_table=monetory_table)  # Merge tables
        return rmf_table
    
    # Method to remove outliers
    def remove_outliers(self, data, columns, lower_threshold=0.05, upper_threshold=0.95):
        data_cleaned = data.copy()
        for col in columns:
            lower_quantile = data[col].quantile(lower_threshold)  # Calculate lower quantile
            upper_quantile = data[col].quantile(upper_threshold)  # Calculate upper quantile
            outlier_rows = (data[col] < lower_quantile) | (data[col] > upper_quantile)  # Identify outlier rows
            num_outliers = outlier_rows.sum()  # Count number of outliers
            if num_outliers > 0:
                data_cleaned = data_cleaned.loc[~outlier_rows]  # Remove outliers
                logging.info(f"Removed {num_outliers} outliers from column '{col}'.")
            else:
                logging.info(f"No outliers found in column '{col}'.")
        data_cleaned.reset_index(drop=True, inplace=True)
        return data_cleaned
    
    # Method to handle outliers
    def outlier(self, X):
        X = self.remove_outliers(X, columns=['recency', 'frequency', 'monetary'])  # Remove outliers
        return X
    
    # Method to perform data wrangling
    def data_wrangling(self, X: pd.DataFrame):
        try:
            data_modified = self.run_data_modification(data=X)  # Run data modification steps
            logging.info("Data Modification Done")
            logging.info("Removing Outliers")
            df_outlier_removed = self.outlier(X=data_modified)  # Remove outliers
            return df_outlier_removed
        
        except Exception as e:
            raise CustomException(e, sys) from e
    
    # Method to fit the transformer
    def fit(self, X, y=None):
        return self
    
    # Method to transform the data
    def transform(self, X: pd.DataFrame, y=None):
        try:
            data_modified = self.data_wrangling(X)  # Perform data wrangling
            logging.info(" Data Wrangaling done ")
            logging.info(f"Original Data  : {X.shape}")
            logging.info(f"Shape Modified Data : {data_modified.shape}")
            return data_modified
        
        except Exception as e:
            raise CustomException(e, sys) from e


# Class for data transformation process
class DataTransformation:
    
    # Constructor
    def __init__(self, data_transformation_config: DataTransformationConfig,
                 data_validation_artifact: DataValidationArtifact):
        try:
            logging.info(f"{'>>'*5} Data Transformation log started {'<<'*5} \n\n")
            self.data_transformation_config = data_transformation_config
            self.data_validation_artifact = data_validation_artifact
            self.transformation_yaml = read_yaml_file(file_path=TRANSFORMATION_YAML_FILE_PATH)  # Read transformation YAML file
            self.drop_columns = self.transformation_yaml[DROP_COLUMNS]  # Get columns to drop from YAML
        
        except Exception as e:
            raise CustomException(e, sys) from e
    
    # Method to get feature engineering object
    def get_feature_engineering_object(self):
        try:
            feature_engineering = Pipeline(steps=[("fe", Feature_Engineering(drop_columns=self.drop_columns))])  # Create feature engineering pipeline
            return feature_engineering
        
        except Exception as e:
            raise CustomException(e, sys) from e
    
    # Method to get data transformer object
    def get_data_transformer_object(self):
        try:
            logging.info('Creating Data Transformer Object')
            numerical_col = ['recency', 'frequency', 'monetary']
            numerical_pipeline = Pipeline(steps=[('scaler', StandardScaler())])
            preprocessor = ColumnTransformer([('numerical_pipeline', numerical_pipeline, numerical_col)])
            return preprocessor
        
        except Exception as e:
            logging.error('An error occurred during data transformation')
            raise CustomException(e, sys) from e
    
    # Method to initiate data transformation process
    def initiate_data_transformation(self):
        try:
            logging.info(f"Obtaining training and test file path.")
            train_file_path = self.data_validation_artifact.validated_train_path
            
            logging.info(f"Loading training and test data as pandas dataframe.")
            train_df = pd.read_csv(train_file_path)  # Load training data
            
            logging.info(f" Accessing train and test data \n Train Data : {train_file_path}")
            logging.info(f" Training columns {train_df.columns}")
            logging.info(f"Obtaining feature engineering object.")
            fe_obj = self.get_feature_engineering_object()  # Get feature engineering object
            
            logging.info(f"Applying feature engineering object on training dataframe and testing dataframe")
            logging.info(">>>" * 5 + " Training data " + "<<<" * 5)
            logging.info(f"Feature Engineering - train data ")
            train_df = fe_obj.fit_transform(train_df)  # Apply feature engineering on training data
            
            logging.info(f"Saving feature engineered training dataframe.")
            transformed_train_dir = self.data_transformation_config.transformed_train_dir
            feature_eng_train_file_path = os.path.join(transformed_train_dir, "feature_engineering.csv")  # Save feature engineered training data
            save_data(file_path=feature_eng_train_file_path, data=train_df)
            
            logging.info("*" * 5 + " Applying preprocessing object on training dataframe  " + "*" * 5)
            preprocessing_obj = self.get_data_transformer_object()  # Get data transformer object
            train_arr = preprocessing_obj.fit_transform(train_df)  # Apply data transformation on training data
            
            logging.info(f"Shape of train_arr: {train_arr.shape}")
            logging.info("Transformation completed successfully")
            col = ['recency', 'frequency', 'monetary']
            
            transformed_train_df = pd.DataFrame(train_arr, columns=col)
            transformed_train_df['CustomerID'] = train_df['CustomerID']
            transformed_train_dir = self.data_transformation_config.transformed_train_dir
            transformed_train_file_path = os.path.join(transformed_train_dir, "transformed_train.csv")  # Save transformed training data
            
            logging.info("Saving Transformed Train file")
            save_data(file_path=transformed_train_file_path, data=transformed_train_df)
            
            logging.info("Transformed Train file saved")
            logging.info("Saving Feature Engineering Object")
            logging.info("Saving Feature Engineering Object")
            feature_engineering_object_file_path = self.data_transformation_config.feature_engineering_object_file_path
            save_object(file_path=feature_engineering_object_file_path, obj=fe_obj)  # Save feature engineering object
            save_object(file_path=os.path.join(ROOT_DIR, PIKLE_FOLDER_NAME_KEY,
                                                os.path.basename(feature_engineering_object_file_path)), obj=fe_obj)
            
            logging.info("Saving Preprocessing Object")
            preprocessing_object_file_path = self.data_transformation_config.preprocessed_object_file_path
            save_object(file_path=preprocessing_object_file_path, obj=preprocessing_obj)  # Save preprocessing object
            save_object(file_path=os.path.join(ROOT_DIR, PIKLE_FOLDER_NAME_KEY,
                                                os.path.basename(preprocessing_object_file_path)), obj=preprocessing_obj)
            feature_eng_train_file_path = feature_eng_train_file_path
            data_transformation_artifact = DataTransformationArtifact(is_transformed=True,
                                                                      message="Data Transformation successfull.",
                                                                      feature_eng_train_file_path=feature_eng_train_file_path,
                                                                      transformed_train_file_path=transformed_train_file_path,
                                                                      preprocessed_object_file_path=preprocessing_object_file_path,
                                                                      feature_engineering_object_file_path=feature_engineering_object_file_path)
            logging.info(f"Data Transformation Artifact: {data_transformation_artifact}")
            return data_transformation_artifact
        
        except Exception as e:
            raise CustomException(e, sys) from e

    # Destructor
    def __del__(self):
        logging.info(f"{'>>'*5} Data Transformation log completed {'<<'*5} \n\n")