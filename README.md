# Cervical--cancer-risk-classification
The project can be found <a href="" target="_blank">here</a>

## Introduction
Cervical cancer is a common type of cancer in women. It can be curable when it is treated in the early stage. Therefore, it is essential for women to be aware of the habits, health conditions, medical history and etc that might lead to cervical cancer.   

This project aims to identify key cervical cancer indicators/predictors and to predict cervical cancer based on demographic information, habits, historical medical records with machine learning algorithms. 

## Table of Content
- [Dataset](#Dataset) 
- [Data Preprocessing](#Data Preprocessing)  
- [Data Visualization](#Data Visualization)  
- [Feature selection](#Feature Selection)  
- [Prediction](#Prediction)
- [Results](#Results)

## Dataset
The data source can be found at [UCI Machine Learning Repository - Cervical cancer (Risk Factors) Data Set](https://archive.ics.uci.edu/ml/datasets/Cervical+cancer+%28Risk+Factors%29) under the name Data Folder. 

## Data Preprocessing
<a href="https://github.com/ellenxxiao/Cervical--cancer-risk-classification/blob/master/Data%20Preprocessing.py" target="_blank">Data preprocessing</a> contains 4 steps:
1. Fill out missing values with correlated features/or median values 
2. Remove missing values
3. Remove unnecessary columns
4. Create features that might have insights and can interpret better

## Data Visualization
<a href="https://github.com/ellenxxiao/Cervical--cancer-risk-classification/blob/master/Data%20Visualization.py" target="_blank">Data visualization</a> section shows the followings:
1. The distributions of features
2. How the disease distributes among features
3. Feature correlations

## Feature Selection
Two models are applied to select features, refer <a href="https://github.com/ellenxxiao/Cervical--cancer-risk-classification/blob/master/Feature%20Selection.py" target="_blank">here</a>
1. RFE
2. RandomForest

## Predictions
Three models are applied to predict cervical cancer, refer <a href="https://github.com/ellenxxiao/Cervical--cancer-risk-classification/blob/master/Prediction.py" target="_blank">here</a>
1. L1-Regularized Logistic Regression
2. L2-Regularized Logistic Regression
3. RandomForest Classification

## Results
Age, Number of pregancies, Hormonal Contraceptives (years), IUD(years), Smokes (years), First secual intercourse age, Smokes(packs/year) are the features appear to be important in both RFE and RandomForest models. 

RandomForest Classification has the best result.
Accuracy: 0.973    
F1-Score: 0.896   
Sensitivity: 0.812

