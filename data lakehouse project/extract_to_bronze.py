import os
import requests
import pandas as pd
from google.cloud import bigquery
from datetime import datetime, timedelta
import time

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "gcp_key.json"
PROJECT_ID = "open-meteo-analytics-platform"

# Define a diverse set of global cities for K-Means climate clustering
CITIES = [
    {"name": "New Delhi", "lat": 28.61, "lon": 77.23},
    {"name": "Mumbai", "lat": 19.07, "lon": 72.87},
    {"name": "London", "lat": 51.51, "lon": -0.13},
    {"name": "New York", "lat": 40.71, "lon": -74.01},
    {"name": "Tokyo", "lat": 35.69, "lon": 139.69},
    {"name": "Sydney", "lat": -33.87, "lon": 151.21},
    {"name": "Cairo", "lat": 30.04, "lon": 31.24},
    {"name": "Reykjavik", "lat": 64.15, "lon": -21.94},
    {"name": "Cape Town", "lat": -33.92, "lon": 18.42},
    {"name": "Rio de Janeiro", "lat": -22.91, "lon": -43.17}
]

def fetch_multi_city_data():
    """Extracts 30 days of weather data for multiple cities."""
    end_date = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
    
    all_data = []
    
    for city in CITIES:
        print(f"Fetching data for {city['name']}...")
        url = (
            f"https://api.open-meteo.com/v1/forecast?"
            f"latitude={city['lat']}&longitude={city['lon']}&"
            f"start_date={start_date}&end_date={end_date}&"
            f"daily=temperature_2m_max,temperature_2m_min,precipitation_sum&"
            f"timezone=auto"
        )
        
        response = requests.get(url)
        response.raise_for_status()
        daily_data = response.json()['daily']
        
        # Build DataFrame for the specific city
        df = pd.DataFrame({
            'city': city['name'],
            'latitude': city['lat'],
            'longitude': city['lon'],
            'date': daily_data['time'],
            'max_temp_celsius': daily_data['temperature_2m_max'],
            'min_temp_celsius': daily_data['temperature_2m_min'],
            'precipitation_mm': daily_data['precipitation_sum']
        })
        all_data.append(df)
        time.sleep(0.5) # Prevent rate-limiting from the free API
        
    final_df = pd.concat(all_data, ignore_index=True)
    final_df['extraction_timestamp'] = pd.Timestamp.utcnow()
    return final_df

def load_to_bronze(df):
    client = bigquery.Client(project=PROJECT_ID)
    table_id = f"{PROJECT_ID}.weather_lakehouse.bronze_weather_data"
    job_config = bigquery.LoadJobConfig(write_disposition="WRITE_APPEND")
    job = client.load_table_from_dataframe(df, table_id, job_config=job_config)
    job.result() 
    print(f"Successfully loaded {len(df)} multi-city rows to Bronze.")

def update_silver_layer():
    client = bigquery.Client(project=PROJECT_ID)
    overwrite_query = """
    CREATE OR REPLACE TABLE `open-meteo-analytics-platform.weather_lakehouse.silver_weather_daily_physical`
    PARTITION BY observation_date AS
    SELECT * EXCEPT(rn)
    FROM (
        SELECT 
            city,
            CAST(latitude AS FLOAT64) AS latitude,
            CAST(longitude AS FLOAT64) AS longitude,
            CAST(date AS DATE) AS observation_date,
            CAST(max_temp_celsius AS FLOAT64) AS max_temp_celsius,
            CAST(min_temp_celsius AS FLOAT64) AS min_temp_celsius,
            CAST(precipitation_mm AS FLOAT64) AS precipitation_mm,
            CAST(extraction_timestamp AS TIMESTAMP) AS extraction_timestamp,
            ROW_NUMBER() OVER(PARTITION BY city, date ORDER BY extraction_timestamp DESC) as rn
        FROM `open-meteo-analytics-platform.weather_lakehouse.bronze_weather_data`
    )
    WHERE rn = 1;
    """
    job = client.query(overwrite_query)
    job.result() 
    print("Successfully deduplicated and updated the Silver layer.")

if __name__ == "__main__":
    multi_city_df = fetch_multi_city_data()
    load_to_bronze(multi_city_df)
    update_silver_layer()