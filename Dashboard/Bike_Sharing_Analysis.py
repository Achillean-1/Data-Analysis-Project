import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

# Import library yang diperlukan
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Judul Aplikasi
st.title("Analisis Data: Bike Sharing Dataset")
st.subheader("Analisis dan Visualisasi Data Penyewaan Sepeda")

# Load Data
@st.cache  # Cache data untuk mempercepat loading
def load_data():
    hour_df = pd.read_csv('Dashboard/hour.csv')
    day_df = pd.read_csv('Dashboard/day.csv')
    return hour_df, day_df

hour_df, day_df = load_data()

# Tampilkan informasi data
if st.checkbox("Tampilkan Informasi Data Hour"):
    st.write(hour_df.info())

if st.checkbox("Tampilkan Informasi Data Day"):
    st.write(day_df.info())

# Tampilkan Deskripsi Data
if st.checkbox("Tampilkan Deskripsi Data Hour"):
    st.write(hour_df.describe())

if st.checkbox("Tampilkan Deskripsi Data Day"):
    st.write(day_df.describe())

# Visualisasi
st.subheader("Visualisasi Penyewaan Sepeda per Jam")
plt.figure(figsize=(10, 5))
sns.lineplot(data=hour_df, x='hr', y='cnt')
plt.title('Jumlah Penyewaan Sepeda per Jam')
plt.xlabel('Jam')
plt.ylabel('Jumlah Penyewaan')
st.pyplot(plt)  # Menampilkan plot dengan Streamlit

# Metrik
st.subheader("Metrik Penyewaan Sepeda")
st.write(f"Total Penyewaan Sepeda: {hour_df['cnt'].sum()}")

# Model Prediksi (Contoh sederhana)
if st.checkbox("Tampilkan Model Prediksi"):
    # Memisahkan fitur dan target
    X = hour_df[['temp', 'hum', 'windspeed']]
    y = hour_df['cnt']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Membuat dan melatih model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Prediksi
    y_pred = model.predict(X_test)
    
    # Evaluasi
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    st.write(f"Mean Squared Error: {mse:.2f}")
    st.write(f"R^2 Score: {r2:.2f}")
