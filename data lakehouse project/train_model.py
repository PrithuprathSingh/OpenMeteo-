import os
import pandas as pd
from google.cloud import bigquery
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
import mlflow
import mlflow.sklearn
import joblib

# 1. Authenticate and Extract Gold Data
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "gcp_key.json"
client = bigquery.Client()

query = """
    SELECT 
        city, latitude, longitude, max_temp_celsius, min_temp_celsius,
        temperature_variance, rolling_3d_avg_max_temp, did_it_rain
    FROM `open-meteo-analytics-platform.weather_lakehouse.gold_weather_features`
"""
df = client.query(query).to_dataframe().dropna()

print("--- 1. Training MLOps Support Vector Machine ---")

# Fix: Force MLflow to use a local SQLite database file to avoid Windows path errors
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("Weather_Precipitation_SVM")

with mlflow.start_run():
    X_svm = df[['max_temp_celsius', 'min_temp_celsius', 'temperature_variance', 'rolling_3d_avg_max_temp']]

with mlflow.start_run():
    X_svm = df[['max_temp_celsius', 'min_temp_celsius', 'temperature_variance', 'rolling_3d_avg_max_temp']]
    y_svm = df['did_it_rain']
    
    X_train, X_test, y_train, y_test = train_test_split(X_svm, y_svm, test_size=0.2, random_state=42)
    
    kernel_type = 'linear'
    svm_model = SVC(kernel=kernel_type, random_state=42)
    svm_model.fit(X_train, y_train)
    
    predictions = svm_model.predict(X_test)
    acc = accuracy_score(y_test, predictions)
    
    # Log hyperparameters and metrics to MLflow
    mlflow.log_param("kernel", kernel_type)
    mlflow.log_metric("accuracy", acc)
    mlflow.sklearn.log_model(svm_model, "svm_model")
    
    joblib.dump(svm_model, 'weather_svm_model.pkl')
    print(f"SVM trained with accuracy: {acc:.2f}. Run logged to MLflow.")

print("\n--- 2. Training Unsupervised K-Means Clustering ---")
# Extract features to group similar physical climate systems
X_kmeans = df[['latitude', 'longitude', 'max_temp_celsius', 'temperature_variance']]

# Initialize K-Means to find 3 distinct climate clusters
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df['climate_cluster'] = kmeans.fit_predict(X_kmeans)

joblib.dump(kmeans, 'climate_kmeans_model.pkl')

# Save a local CSV of the clustered data so our Streamlit app can plot the map instantly
df.to_csv("clustered_weather_data.csv", index=False)
print("K-Means clustering complete. Cities grouped into 3 atmospheric clusters.")