# Customer Segmentation using RFM Analysis

## Overview:

- The project aimed to analyze customer data and develop customer segmentation based on purchasing power and income levels using RFM analysis and machine learning models. This approach enabled the business to understand the spending behavior and purchasing power of different customer segments, thereby optimizing marketing efforts and increasing revenue. The project utilized clustering algorithms such as K-means Clustering,  Gaussian Mixture Model (GMM) for segmentation, as well as a comprehensive data pipeline for preprocessing, validation, transformation, and model training.

## Methodologies:

**1. Data Ingestion and Validation**: Ingested customer data and performed data validation checks, including file name, column labels, data types, and missing values validation.

**2. Data Transformation and Feature Engineering**: Transformed and engineered features from the data to enhance model performance, including creating new features based on insights from exploratory data analysis (EDA).

**3. Model Training and Evaluation**: Utilized clustering algorithms like K-means Clustering and Gaussian Mixture Model (GMM) to segment customers based on income and purchasing power and evaluated models using metrics such as the silhouette score.

**4. Model Pushing**: Saved the optimal models and associated reports for future use, ensuring the best-performing models were available for batch predictions.

**5. UI for User Interaction**: Developed a user-friendly interface for stakeholders to interact with trained models and cluster labeling.

## Technologies and Tools Used:

**1. Packages**: pandas, numpy, matplotlib, seaborn, scipy, scikit-learn.

**2. Machine Learning Algorithms**: Clustering algorithms such as K-means Clustering and Gaussian Mixture Model (GMM).

## Outcomes:

**1. Customer Segmentation**: Identified key customer segments such as "**Affluent Customers**" (high income, high spending), "**Middle-Income Customers**" (moderate income and spending), and "**Budget-Conscious Customers**" (low income, low spending).

**2. Enhanced Marketing Strategies**: Provided actionable insights for targeted marketing efforts, enabling the business to develop customized strategies for each customer segment.

**3. Optimized Data Pipeline**: Implemented a robust data pipeline that included data ingestion, validation, transformation, and model training, ensuring high data quality and consistency for model development.

**4. Model Selection and Deployment**: Trained and evaluated machine learning models, saving the best-performing models for future predictions and batch predictions.
