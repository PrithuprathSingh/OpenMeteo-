import streamlit as st
from google.cloud import bigquery
import os
import pandas as pd

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "gcp_key.json"

st.set_page_config(page_title="Weather Lakehouse", page_icon="🌤️", layout="wide")
st.title("🌤️ Advanced Weather Lakehouse & MLOps")
st.markdown("An end-to-end Medallion pipeline serving live BigQuery features and Machine Learning models.")

@st.cache_data(ttl=3600)
def load_bq_data():
    client = bigquery.Client()
    query = """
        SELECT * FROM `open-meteo-analytics-platform.weather_lakehouse.gold_weather_features`
        ORDER BY observation_date DESC
    """
    return client.query(query).to_dataframe()

df_bq = load_bq_data()

# KPI Metrics
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Records", len(df_bq))
col2.metric("Global Max Temp", f"{df_bq['max_temp_celsius'].max()} °C")
col3.metric("Global Min Temp", f"{df_bq['min_temp_celsius'].min()} °C")
col4.metric("Cities Tracked", df_bq['city'].nunique())

st.divider()

# Unsupervised Learning Overview
st.subheader("🌍 K-Means Climate Clustering")
st.markdown("Cities partitioned into 3 distinct atmospheric groups based on temperature dynamics.")

try:
    df_clusters = pd.read_csv("clustered_weather_data.csv")
    color_map = {0: '#ff4b4b', 1: '#00fa9a', 2: '#1e90ff'}
    df_clusters['color'] = df_clusters['climate_cluster'].map(color_map)
    st.map(df_clusters, latitude="latitude", longitude="longitude", color="color", zoom=1)
except FileNotFoundError:
    st.warning("Cluster data not found. Please run train_model.py first.")