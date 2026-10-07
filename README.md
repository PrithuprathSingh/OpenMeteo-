An enterprise-grade, end-to-end Data Lakehouse and Machine Learning Operations (MLOps) pipeline. The system automates historical weather extraction across 10 global cities, enforces data idempotency using a partitioned Medallion architecture in Google Cloud BigQuery, manages model governance via an MLflow tracking server, and serves multi-dimensional algorithmic diagnostics through a multi-page interactive Streamlit dashboard.

---

## 🏛️ System Architecture
[ Open-Meteo REST API ]
                                      │
                            (Python Ingestion)
                                      ▼
            ┌──────────────────────────────────────────────────┐
            │        Google Cloud BigQuery Lakehouse           │
            │                                                  │
            │   [Bronze Layer] : Raw append-only JSON staging   │
            │                           │                      │
            │                           ▼ (Window Deduplication)│
            │   [Silver Layer] : Partitioned physical tables   │
            │                           │                      │
            │                           ▼ (Feature Transform)  │
            │   [Gold Layer]   : Feature Store SQL view        │
            └───────────────────────────┬──────────────────────┘
                                        │
                  ┌─────────────────────┴─────────────────────┐
                  ▼                                           ▼
      [ Supervised Learning ]                     [ Unsupervised Learning ]
    Support Vector Machine (SVM)                      K-Means Clustering
   (Precipitation Classification)                 (Global Climate Profiling)
                  │                                           │
                  ▼                                           ▼
         [ MLflow Tracking ]                          [ Serialization ]
      (SQLite Backend Registry)                     (.pkl Model Artifacts)
                  │                                           │
                  └─────────────────────┬─────────────────────┘
                                        ▼
                ┌───────────────────────────────────────────┐
                │     Multi-Page Streamlit Web App          │
                │                                           │
                │  ├─ app.py (Executive KPI & Cluster Map)  │
                │  ├─ 1_Algorithmic_Diagnostics.py          │
                │  │   (3D Decision Space & Batch Inference)│
                │  └─ 2_Physical_Earth_Systems.py           │
                │      (PyDeck 3D Variance Columns & TS)    │
                └───────────────────────────────────────────┘
## 🚀 Key Architectural Features

### 1. Medallion Lakehouse Architecture (BigQuery)
* **Bronze Layer (`bronze_weather_data`)**: Append-only landing zone capturing raw telemetry (`date`, `max_temp`, `min_temp`, `precipitation`, `city`, coordinates, and ingestion timestamps).
* **Silver Layer (`silver_weather_daily_physical`)**: Partitioned by `observation_date` with built-in idempotency. It applies an analytical `ROW_NUMBER() OVER(PARTITION BY city, date ORDER BY extraction_timestamp DESC)` filter to ensure zero duplicate records even when pipeline runs overlap.
* **Gold Layer (`gold_weather_features`)**: Analytical feature store generating machine learning signals:
  * Binary classification target: `did_it_rain` (`precipitation_mm > 0`).
  * Atmospheric variance: `temperature_variance` ($T_{max} - T_{min}$).
  * Localized thermal trends: `rolling_3d_avg_max_temp` partitioned strictly by city.

### 2. MLOps Lifecycle & Governance (MLflow)
* Centralized experiment tracking utilizing a local SQLite backend to prevent OS path serialization collisions.
* Automated logging of model hyperparameters (`kernel="linear"`), evaluation metrics (`accuracy_score`), and Scikit-Learn model artifacts directly to the model registry.

### 3. Dual-Paradigm Machine Learning
* **Supervised Classification (SVM)**: Support Vector Classifier mapping atmospheric feature vectors to separate clear and rainy conditions along a maximum-margin hyperplane.
* **Unsupervised Clustering (K-Means)**: Groups global cities into 3 macro-climates based on coordinate geography, maximum heat, and temperature variance vectors.

### 4. Interactive Multi-Page Analytical Web Application
* **Executive Hub (`app.py`)**: Top-level operational metrics, cached queries (`ttl=3600`), and a global map of K-Means climate assignments.
* **Algorithmic Diagnostics (`1_Algorithmic_Diagnostics.py`)**: High-resolution 3D Plotly projection of SVM feature space with categorical separation, accompanied by a batch CSV processing interface.
* **Physical Earth Systems (`2_Physical_Earth_Systems.py`)**: 3D PyDeck column map scaling elevation to thermal variance with custom RGB heat gradients, alongside multi-city rolling average time-series visualizations.

---

## 🛠️ Tech Stack & Prerequisites

* **Language**: Python 3.10+
* **Cloud Platform**: Google Cloud Platform (BigQuery Sandbox or Standard)
* **Data Manipulation**: Pandas, PyArrow
* **Machine Learning**: Scikit-Learn, Joblib
* **Experiment Tracking**: MLflow (SQLite storage backend)
* **Serving & Geospatial**: Streamlit, PyDeck, Plotly Express
* **APIs**: Open-Meteo Historical Weather API

---

