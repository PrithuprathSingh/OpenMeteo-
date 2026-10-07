import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import os

st.set_page_config(page_title="Algorithmic Diagnostics", page_icon="🤖", layout="wide")
st.title("🤖 Algorithmic Diagnostics & Prediction")

@st.cache_resource
def load_model():
    model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'weather_svm_model.pkl')
    return joblib.load(model_path)

try:
    svm_model = load_model()
    
    st.subheader("Support Vector Machine: Feature Space Geometry")
    st.markdown("Visualizing the multi-dimensional boundaries optimized by the linear kernel.")
    
    data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'clustered_weather_data.csv')
    df = pd.read_csv(data_path)
    
    # 1. NEW FIX: Convert the binary target into a categorical string for discrete plotting
    df['Precipitation_Class'] = df['did_it_rain'].map({1: 'Rain (1)', 0: 'Clear (0)'})
    
    # 2. UPDATED PLOT: Use distinct colors and apply the categorical class
    fig = px.scatter_3d(
        df, 
        x='max_temp_celsius', 
        y='min_temp_celsius', 
        z='temperature_variance',
        color='Precipitation_Class', 
        symbol='Precipitation_Class',
        color_discrete_map={'Rain (1)': '#00BFFF', 'Clear (0)': '#FFD700'},
        title="3D Projection of Precipitation Classification"
    )
    
    # 3. NEW FIX: Shrink the markers, add transparency, and expand the layout height
    fig.update_traces(marker=dict(size=4, opacity=0.8, line=dict(width=0.5, color='DarkSlateGrey')))
    fig.update_layout(height=700, margin=dict(l=0, r=0, b=0, t=40))
    
    st.plotly_chart(fig, use_container_width=True)

    st.divider()
    
    st.subheader("Batch Inference Pipeline")
    st.markdown("Upload a matrix of atmospheric features for automated SVM classification.")
    
    uploaded_file = st.file_uploader("Upload CSV matrix", type="csv")
    if uploaded_file is not None:
        input_df = pd.read_csv(uploaded_file)
        required_cols = ['max_temp_celsius', 'min_temp_celsius', 'temperature_variance', 'rolling_3d_avg_max_temp']
        
        if all(col in input_df.columns for col in required_cols):
            predictions = svm_model.predict(input_df[required_cols])
            input_df['Precipitation_Forecast'] = ["Rain" if p == 1 else "Clear" for p in predictions]
            st.success("Batch array processing complete. Model converged successfully.")
            st.dataframe(input_df, use_container_width=True)
        else:
            st.error(f"Schema mismatch. Required columns: {required_cols}")

except FileNotFoundError:
    st.error("Model artifacts missing. Ensure 'weather_svm_model.pkl' exists in the root directory.")