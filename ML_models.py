import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet

# Perform PCA on the transposed data to reduce dimensionality.
def perform_pca(data, n_components=10):
    pca = PCA(n_components=n_components)
    reduced_data = pca.fit_transform(data.T)
    return reduced_data

# Perform K-means clustering on reduced data.
def kmeans_clustering(data, n_clusters=4):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(data)
    return clusters

# Fit Prophet model on training data
def fit_prophet_model(train_data):
    df_prophet = train_data.reset_index()
    df_prophet.columns = ['ds', 'y']
    prophet_model = Prophet()
    prophet_model.fit(df_prophet)
    return prophet_model


