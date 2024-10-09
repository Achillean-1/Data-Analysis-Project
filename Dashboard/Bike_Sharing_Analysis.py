import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

st.title("Analisis Data: Bike Sharing Dataset")
st.subheader("Analisis dan Visualisasi Data Penyewaan Sepeda")

@st.cache_data
def load_data():
    hour_df = pd.read_csv('Dashboard/hour.csv')
    day_df = pd.read_csv('Dashboard/day.csv')
    return hour_df, day_df

hour_df, day_df = load_data()

if st.checkbox("Tampilkan Informasi Data Hour"):
    st.write(hour_df.info())

if st.checkbox("Tampilkan Informasi Data Day"):
    st.write(day_df.info())

if st.checkbox("Tampilkan Deskripsi Data Hour"):
    st.write(hour_df.describe())

if st.checkbox("Tampilkan Deskripsi Data Day"):
    st.write(day_df.describe())

st.subheader("Visualisasi Penyewaan Sepeda per Jam")
fig, ax = plt.subplots(figsize=(10, 5))
sns.lineplot(data=hour_df, x='hr', y='cnt', ax=ax)
ax.set_title('Jumlah Penyewaan Sepeda per Jam')
ax.set_xlabel('Jam')
ax.set_ylabel('Jumlah Penyewaan')
st.pyplot(fig)

st.subheader("Visualisasi Penyewaan Sepeda Berdasarkan Musim")
fig, ax = plt.subplots(figsize=(10, 5))
sns.countplot(x='season', data=hour_df, ax=ax)
ax.set_title('Distribusi Penyewaan Sepeda Berdasarkan Musim')
ax.set_xlabel('Musim')
ax.set_ylabel('Frekuensi')
st.pyplot(fig)

st.subheader("Heatmap Korelasi")
numeric_df = hour_df.select_dtypes(include=['float64', 'int64'])
corr_matrix = numeric_df.corr()
fig, ax = plt.subplots(figsize=(14, 10))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', ax=ax)
ax.set_title('Correlation Heatmap')
st.pyplot(fig)

st.subheader("RFM Analysis")
day_df['dteday'] = pd.to_datetime(day_df['dteday'])
latest_date = day_df['dteday'].max()

rfm = day_df.groupby('dteday').agg({
    'cnt': 'sum',
    'casual': 'sum',
    'registered': 'sum'
}).reset_index()

recency_df = day_df[['dteday']].drop_duplicates()
recency_df['Recency'] = (latest_date - recency_df['dteday']).dt.days
rfm = rfm.merge(recency_df, on='dteday')
rfm.columns = ['dteday', 'Frequency', 'Monetary', 'Registered', 'Recency']

rfm['R_Score'] = pd.qcut(rfm['Recency'], 4, labels=[4, 3, 2, 1])
rfm['F_Score'] = pd.qcut(rfm['Frequency'], 4, labels=[1, 2, 3, 4])
rfm['M_Score'] = pd.qcut(rfm['Monetary'], 4, labels=[1, 2, 3, 4])
rfm['RFM_Score'] = rfm[['R_Score', 'F_Score', 'M_Score']].sum(axis=1)

st.write("RFM Analysis:")
st.write(rfm.head())

hour_df['usage_level'] = pd.cut(hour_df['hr'], bins=[0, 6, 12, 18, 24], labels=['low', 'medium', 'high', 'very high'])
usage_distribution = hour_df.groupby('usage_level')['cnt'].sum().reset_index()
st.write("Distribusi Penyewaan Berdasarkan Tingkat Penggunaan:")
st.write(usage_distribution)

if st.checkbox("Tampilkan Model Prediksi"):
    X = hour_df[['temp', 'hum', 'windspeed']]
    y = hour_df['cnt']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Prediksi dan evaluasi model
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # Menampilkan hasil evaluasi
    st.write(f"Mean Squared Error: {mse:.2f}")
    st.write(f"R^2 Score: {r2:.2f}")

