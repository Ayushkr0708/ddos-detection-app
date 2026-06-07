# Project Report: Advanced DDoS Detection System

## Objective
The objective of this project is to detect Distributed Denial of Service (DDoS) attacks from network traffic data using machine learning.

## Dataset
The dataset is stored at `datasets/dataset.csv` and contains CICIDS-style flow features with a target label column.

## Methodology
1. Data loading and cleaning
2. Infinite and missing value handling
3. Numeric feature selection
4. Feature scaling with MinMaxScaler
5. Ensemble model training using Random Forest, ExtraTrees, and HistGradientBoosting
6. Evaluation using accuracy, precision, recall, F1 score, ROC-AUC, and average precision
7. Streamlit dashboard deployment

## Frontend Features
- Animated dashboard
- CSV upload and prediction
- BENIGN/DDoS prediction tables
- Confidence score analysis
- Risk level distribution
- Probability curves
- Feature importance visualization
- Correlation analysis
- Downloadable prediction report

## Deployment
The app is designed for Streamlit Community Cloud deployment through GitHub.
