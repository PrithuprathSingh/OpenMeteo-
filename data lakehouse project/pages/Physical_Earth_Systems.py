import streamlit as st
from google.cloud import bigquery
import pydeck as pdk
import os
import plotly.express as px
import pandas as pd

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "gcp_key.json"

st.set_page_config(page_title="Physical Earth Systems", page_icon="🌍", layout="wide")
st.title("🌍 Physical Earth Systems & Dynamics")

@st.cache_data(ttl=3600)
def load_geospatial_data():
    client = bigquery.Client()
    # Removed the ARRAY_AGG(STRUCT) column that PyDeck was unable to serialize
    query = """
        SELECT city, latitude, longitude, AVG(temperature_variance) as avg_variance, 
               MAX(max_temp_celsius) as absolute_max
        FROM `open-meteo-analytics-platform.weather_lakehouse.gold_weather_features`
        GROUP BY city, latitude, longitude
    """
    return client.query(query).to_dataframe()

df_geo = load_geospatial_data()

# Calculate the dynamic 3D column colors mathematically in Pandas
# Warmer climates scale higher in Red [R], colder climates scale higher in Blue [B]
df_geo['color'] = df_geo['absolute_max'].apply(
    lambda x: [min(int(x * 6), 255), 50, max(255 - int(x * 3), 0), 200]
)

st.subheader("Global Atmospheric Variance")
st.markdown("3D geospatial rendering of temperature volatility across regions.")

# PyDeck 3D Column Layer mapping variance to physical height
column_layer = pdk.Layer(
    "ColumnLayer",
    data=df_geo,
    get_position=["longitude", "latitude"],
    get_elevation="avg_variance",
    elevation_scale=50000,
    radius=50000,
    get_fill_color="color", 
    pickable=True,
    auto_highlight=True,
)

tooltip = {
    "html": "<b>City:</b> {city} <br/> <b>Avg Variance:</b> {avg_variance}°C <br/> <b>Max Temp:</b> {absolute_max}°C",
    "style": {"background": "grey", "color": "white", "font-family": '"Helvetica Neue", Arial', "z-index": "10000"}
}

r = pdk.Deck(
    layers=[column_layer],
    initial_view_state=pdk.ViewState(latitude=20.0, longitude=0.0, zoom=1, pitch=45, bearing=10),
    tooltip=tooltip,
    map_style=pdk.map_styles.DARK
)
st.pydeck_chart(r)

st.divider()

st.subheader("Dynamic 3-Day Rolling Averages")
st.markdown("Tracking shifting thermal masses and weather fronts over time.")

# Execute a separate, clean query for the Plotly time-series chart
client = bigquery.Client()
query_ts = "SELECT city, observation_date, rolling_3d_avg_max_temp FROM `open-meteo-analytics-platform.weather_lakehouse.gold_weather_features`"
df_ts = client.query(query_ts).to_dataframe()

fig_ts = px.line(df_ts, x="observation_date", y="rolling_3d_avg_max_temp", color="city", markers=True)
st.plotly_chart(fig_ts, use_container_width=True)