import sys, os
from src.logger import logging
from src.exception import CustomException
from src.utils.common import save_object,read_yaml_file
from src.entity.config_entity import *
from src.entity.artifact_entity import *
from src.constants import *
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
import yaml
import shutil


# Definition of the Model class
class Model:
    
    # Constructor to initialize model object with model_png_location
    def __init__(self, model_png_location):
        
        self.model_png_location = model_png_location  # Assigning model_png_location attribute
        file_location = self.model_png_location  # Assigning model_png_location to file_location
        os.makedirs(file_location, exist_ok=True)  # Creating directory if not exists
    
    # Method to choose optimal number of clusters using silhouette score
    def choose_clusters(self, df, max_clusters=50):
        silhouette_scores = []  # List to store silhouette scores
        
        try:
            for k in range(2, max_clusters+1):  # Looping through range of clusters
                kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42)  # Creating KMeans instance
                kmeans.fit(df)  # Fitting KMeans model
                labels = kmeans.predict(df)  # Predicting labels
                silhouette_avg = silhouette_score(df, labels)  # Calculating silhouette score
                silhouette_scores.append(silhouette_avg) 
        
        except Exception as e:  # Catching exceptions
            raise CustomException(e, sys) from e
        
        optimal_clusters = np.argmax(silhouette_scores) + 2  # Finding index of max silhouette score
        
        return optimal_clusters
    
    # Method to perform KMeans clustering
    def perform_kmeans_clustering(self, df, n_clusters):
        kmeans = KMeans(n_clusters=n_clusters, init='k-means++', algorithm='lloyd')  # Creating KMeans instance
        kmeans.fit(df)  # Fitting KMeans model
        labels = kmeans.predict(df)  # Predicting labels
        centroids = kmeans.cluster_centers_  # Getting cluster centroids
        
        rfm_kmean = df.copy()
        
        rfm_kmean['cluster'] = labels  # Adding cluster labels to DataFrame
        logging.info(" Kmeans Fitted ")
        
        return rfm_kmean, kmeans
    
    # Method to save RFM plot
    def save_rfm_plot(self, rfm_df, directory, model_name, cluster_column, fea_eng_data):
        # Extracting relevant columns from feature engineered data
        rfm_df[['recency', 'frequency', 'monetary']] = fea_eng_data[['recency', 'frequency', 'monetary']]
        cluster_labels = rfm_df[cluster_column]  # Getting cluster labels

        num_clusters = len(cluster_labels.unique())  # Counting unique clusters
        colors = sns.color_palette('viridis', num_clusters)  # Generating colors for clusters

        fig, axes = plt.subplots(1, 1, figsize=(8, 6))  # Creating figure and axes

        # Plotting clusters
        for cluster_label, color in zip(cluster_labels.unique(), colors):
            cluster_data = rfm_df[cluster_labels == cluster_label]
            axes.scatter(
                cluster_data['frequency'],
                cluster_data['monetary'],
                color=color,
                label=f'{cluster_label}'
            )

        axes.set_xlabel('Frequency')  # Setting x-axis label
        axes.set_ylabel('Monetary')  # Setting y-axis label
        axes.set_title('Frequency vs Monetary')  # Setting plot title
        axes.legend()  # Adding legend

        plt.tight_layout()  # Adjusting layout

        file_path = os.path.join(directory, model_name)  # Creating file path
        os.makedirs(file_path, exist_ok=True)  # Creating directory if not exists

        filename = os.path.join(file_path, 'prediction.png')  # Creating filename
        fig.savefig(filename)

        return filename
    
    # Method to train KMeans model
    def Kmeans_train(self, data):
        optimal_cluster = 4  # Optimal cluster based on EDA
        logging.info(f" Based on silhouette_score optimal number of clusters ---K_MEANS--- : {optimal_cluster}")
        
        rfm_kmean, kmeans_model = self.perform_kmeans_clustering(df=data, n_clusters=optimal_cluster)  # Performing KMeans clustering
        
        return rfm_kmean, kmeans_model, optimal_cluster
    
    # Method to find optimal clusters using Gaussian Mixture Model
    def find_optimal_clusters(self, data, max_clusters):
        if max_clusters < 3:  # Checking if max_clusters is less than 3
            raise ValueError("max_clusters must be at least 3")

        silhouette_scores = []  # List to store silhouette scores

        for k in range(3, max_clusters + 1):  # Looping through range of clusters
            model = GaussianMixture(n_components=k, init_params='k-means++')  # Creating Gaussian Mixture Model instance
            model.fit(data)  # Fitting GMM model
            labels = model.predict(data)  # Predicting labels
            score = silhouette_score(data, labels)  # Calculating silhouette score
            silhouette_scores.append(score)

        optimal_clusters, _ = max(enumerate(silhouette_scores), key=lambda x: x[1])  # Finding index of max silhouette score

        logging.info(f"Optimal Clusters in GMM: {optimal_clusters + 3}")

        return optimal_clusters + 3
    
    # Method to perform Gaussian Mixture Model clustering
    def GaussianMixtureClustering(self, data, optimal_clusters):
        model = GaussianMixture(n_components=optimal_clusters, init_params='k-means++', covariance_type='spherical')  # Creating GMM instance
        model.fit(data)  # Fitting GMM model
        labels = model.predict(data)  # Predicting labels

        logging.info("Labels created")

        return labels, model
    
    # Method to add cluster labels to data
    def adding_labels_to_data(self, data, cluster_labels):
        logging.info("Converting cluster_labels to DataFrame")
        df_cluster_labels = pd.DataFrame(cluster_labels, columns=['cluster'])  # Converting cluster labels to DataFrame
        data['cluster'] = df_cluster_labels['cluster']  # Adding cluster labels to DataFrame
        logging.info("Cluster added to the DataFrame")

        return data
    
    # Method to train Gaussian Mixture Model
    def GaussianMixtureClusteringTrain(self, data):
        optimal_cluster = self.find_optimal_clusters(data=data, max_clusters=4)  # Finding optimal number of clusters

        logging.info("Performing Gaussian Mixture Model clustering")
        labels, gmm_model = self.GaussianMixtureClustering(data=data, optimal_clusters=optimal_cluster)  # Performing GMM clustering

        logging.info("Adding cluster labels to the data")
        data = self.adding_labels_to_data(data=data, cluster_labels=labels)  # Adding cluster labels to data

        rfm_gmm_table = data  # Assigning clustered DataFrame to rfm_gmm_table

        return rfm_gmm_table, gmm_model, optimal_cluster
    
# Definition of the ModelTrainer class
class ModelTrainer:

    # Constructor to initialize ModelTrainer object
    def __init__(self, model_trainer_config: ModelTrainerConfig, data_transformation_artifact: DataTransformationArtifact):
        try:
            logging.info(f"{'>>' * 5}Model trainer log started.{'<<' * 5} ")
            
            # Assigning attributes
            self.model_trainer_config = model_trainer_config
            self.data_transformation_artifact = data_transformation_artifact

            # Reading schema data from YAML file
            self.schema_data = read_yaml_file(SCHEMA_FILE_PATH)
            # Assigning model config path
            self.model_config_path = self.model_trainer_config.model_config_path
        except Exception as e:
            raise CustomException(e, sys) from e
    
    # Method to compare cluster labels and select best model
    def compare_cluster_labels(self, kmeans_table, gmm_table, data, optimal_cluster_kmean, optimal_cluster_gmm,
                               k_means_model, kmean_prediction_png,
                               gmm_model, gmm_prediction_png):

        kmeans_labels = kmeans_table['cluster']  # Getting KMeans cluster labels
        gmm_labels = gmm_table['cluster']  # Getting GMM cluster labels

        silhouette_kmeans = silhouette_score(data, kmeans_labels)  # Calculating silhouette score for KMeans
        silhouette_gmm = silhouette_score(data, gmm_labels)  # Calculating silhouette score for GMM

        # Comparing silhouette scores and selecting best model
        if silhouette_kmeans >= silhouette_gmm:
            return 'K-means', k_means_model, silhouette_kmeans, kmean_prediction_png, optimal_cluster_kmean, kmeans_table
        else:
            return 'GMM', gmm_model, silhouette_gmm, gmm_prediction_png, optimal_cluster_gmm, gmm_table
    
    # Method to initiate model training
    def initiate_model_training(self) -> ModelTrainerArtifact:
        try:
            logging.info("Finding transformed Training data")
            transformed_train_file_path = self.data_transformation_artifact.transformed_train_file_path  # Getting transformed train file path
            
            logging.info("Transformed Data found!!! Now, converting it into dataframe")
            train_df = pd.read_csv(transformed_train_file_path)  # Reading transformed train data into DataFrame

            train_rfm_table = train_df[['recency', 'frequency', 'monetary']]  # Extracting relevant columns from train data
            train_data = self.data_transformation_artifact.feature_eng_train_file_path  # Getting feature engineered train data path
            train_data_df = pd.read_csv(train_data)  # Reading feature engineered train data into DataFrame

            logging.info(" Training Kmeans.....")
            model = Model(model_png_location=self.model_trainer_config.png_location)  # Creating Model instance

            # Training KMeans model
            rfm_kmean_table, kmeans_model, optimal_cluster_kmean = model.Kmeans_train(data=train_rfm_table)
            
            # Saving KMeans clustering plot
            prediction_png_kmeans = model.save_rfm_plot(rfm_df=rfm_kmean_table, model_name='k_mean_prediction_data',
                                                        directory=self.model_trainer_config.png_location,
                                                        cluster_column='cluster', fea_eng_data=train_data_df)
            
            logging.info(" Training GMM Clustering.....")

            # Training GMM clustering model
            rfm_gmm_table, gmm_model, optimal_cluster_gmm = model.GaussianMixtureClusteringTrain(data=train_rfm_table)

            # Saving GMM clustering plot
            prediction_png_gmm = model.save_rfm_plot(rfm_df=rfm_gmm_table, directory=self.model_trainer_config.png_location,
                                                     model_name='gmm_cluster_prediction_data', cluster_column='cluster',
                                                     fea_eng_data=train_data_df)

            logging.info(" Evaluating .....")
            
            # Comparing cluster labels and selecting best model
            model_name, model, silhouette_score, prediction_png_path, optimal_cluster, rfm_table = self.compare_cluster_labels(
                                                                                                        kmeans_table=rfm_kmean_table, gmm_table=rfm_gmm_table,
                                                                                                        k_means_model=kmeans_model, kmean_prediction_png=prediction_png_kmeans,
                                                                                                        optimal_cluster_kmean=optimal_cluster_kmean, gmm_prediction_png=prediction_png_gmm,
                                                                                                        optimal_cluster_gmm=optimal_cluster_gmm, gmm_model=gmm_model, data=train_rfm_table)

            logging.info(f" Model Selected :{model_name}")
            logging.info(f"-------------")

            # Adding CustomerID and RFM columns to clustered DataFrame
            rfm_table['CustomerID'] = train_df['CustomerID']
            rfm_table[['recency', 'frequency', 'monetary']] = train_data_df[['recency', 'frequency', 'monetary']]

            trained_model_directory = self.model_trainer_config.trained_model_directory  # Getting trained model directory
            os.makedirs(trained_model_directory, exist_ok=True)  # Creating directory if not exists
            
            logging.info(f"Saving rfm csv at path: {trained_model_directory}")
            csv_file_path = os.path.join(trained_model_directory, 'rfm.csv')  # Creating CSV file path
            
            logging.info(f" rfm table Columns : {rfm_table.columns}")
            rfm_table.to_csv(csv_file_path)  # Saving clustered DataFrame to CSV file

            trained_model_file_path = self.model_trainer_config.trained_model_file_path  # Getting trained model file path
            logging.info(f"Saving model at path: {trained_model_file_path}")
            save_object(file_path=trained_model_file_path, obj=model)  # Saving model to file

            best_model_name = model_name  # Assigning best model name
            logging.info(f"Saving metrics of model  : {best_model_name}")
            
            # Creating report of trained model
            report = {
                "Model_name": best_model_name,
                "Silhouette_score": str(silhouette_score),
                "number_of_clusters": str(optimal_cluster)
            }

            logging.info(f"Dumping Metrics in report.....")

            model_artifact_report_path = self.model_trainer_config.report_path  # Getting model artifact report path
            report_file_path = os.path.join(model_artifact_report_path, 'report.yaml')  # Creating report file path
            os.makedirs(model_artifact_report_path, exist_ok=True)  # Creating directory if not exists

            # Writing report to YAML file
            with open(report_file_path, 'w') as file:
                yaml.safe_dump(report, file)

            logging.info("Report created")

            # Copying prediction PNG file to desired directory
            trained_model_directory = self.model_trainer_config.trained_model_directory  # Getting trained model directory
            shutil.copy(prediction_png_path, trained_model_directory)

            logging.info(" Copied prediction png file to the desired directory")  # Logging info

            # Creating model trainer artifact
            model_trainer_artifact = ModelTrainerArtifact(
                is_trained=True,
                message="Model Trained successfully",
                model_selected=trained_model_file_path,
                model_name=model_name,
                report_path=report_file_path,
                model_prediction_png=prediction_png_path,
                csv_file_path=csv_file_path
            )

            logging.info(f"Model Trainer Artifact: {model_trainer_artifact}")
            return model_trainer_artifact

        except Exception as e:
            raise CustomException(e, sys) from e

    # Destructor method
    def __del__(self):
        logging.info(f"{'>>' * 5}Model trainer log completed.{'<<' * 5} ")