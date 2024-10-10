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

@st.cache
def load_data():
    hour_df = pd.read_csv('Dashboard/hour.csv')
    day_df = pd.read_csv('Dashboard/day.csv')
    return hour_df, day_df

hour_df, day_df = load_data()

# Show dataset information
if st.checkbox("Tampilkan Informasi Data Hour"):
    st.write(hour_df.info())

if st.checkbox("Tampilkan Informasi Data Day"):
    st.write(day_df.info())

# Data Description
if st.checkbox("Tampilkan Deskripsi Data Hour"):
    st.write(hour_df.describe())

if st.checkbox("Tampilkan Deskripsi Data Day"):
    st.write(day_df.describe())

### EDA Univariate ###
st.subheader("Univariate Analysis: Distribusi Variabel Numerikal")
numeric_columns = hour_df.select_dtypes(include=np.number).columns
fig, ax = plt.subplots(figsize=(12, 8))
hour_df[numeric_columns].hist(ax=ax, bins=20)
plt.suptitle('Distribusi Variabel Numerikal', size=16)
st.pyplot(fig)

### EDA Bivariate ###
st.subheader("Penyewaan Sepeda Berdasarkan Jam dalam Sehari")
plt.figure(figsize=(12, 6))
sns.lineplot(x='hr', y='cnt', data=hour_df, marker="o")
plt.title('Penyewaan Sepeda Berdasarkan Jam dalam Sehari')
plt.xlabel('Jam')
plt.ylabel('Jumlah Penyewaan')
st.pyplot(plt)

st.subheader("Hubungan antara Jumlah Penyewaan dan Temperatur")
plt.figure(figsize=(8, 6))
sns.scatterplot(x='temp', y='cnt', data=hour_df)
plt.title('Penyewaan Sepeda vs Temperatur')
st.pyplot(plt)

### Multivariate Analysis ###
st.subheader("Multivariate Analysis: Korelasi Variabel Numerikal")
numeric_df = hour_df.select_dtypes(include=['float64', 'int64'])
corr_matrix = numeric_df.corr()
plt.figure(figsize=(14, 10))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap')
st.pyplot(plt)

### Categorical vs Numerical Analysis ###
st.subheader("Penyewaan Sepeda Berdasarkan Hari dalam Seminggu")
plt.figure(figsize=(12, 6))
sns.boxplot(x='weekday', y='cnt', data=hour_df)
plt.title('Penyewaan Sepeda Berdasarkan Hari dalam Seminggu')
st.pyplot(plt)

st.subheader("Distribusi Penyewaan Sepeda Berdasarkan Musim")
plt.figure(figsize=(10, 5))
sns.countplot(x='season', data=hour_df)
plt.title('Distribusi Penyewaan Sepeda Berdasarkan Musim')
st.pyplot(plt)

### RFM Analysis ###
st.subheader("RFM Analysis")
latest_date = pd.to_datetime(day_df['dteday']).max()
day_df['dteday'] = pd.to_datetime(day_df['dteday'])
rfm = day_df.groupby('dteday').agg({'cnt': 'sum', 'casual': 'sum', 'registered': 'sum'}).reset_index()

recency_df = day_df[['dteday']].drop_duplicates()
recency_df['Recency'] = (latest_date - recency_df['dteday']).dt.days
rfm = rfm.merge(recency_df, on='dteday')
rfm.columns = ['dteday', 'Frequency', 'Monetary', 'Registered', 'Recency']

rfm['R_Score'] = pd.qcut(rfm['Recency'], 4, labels=[4, 3, 2, 1])
rfm['F_Score'] = pd.qcut(rfm['Frequency'], 4, labels=[1, 2, 3, 4])
rfm['M_Score'] = pd.qcut(rfm['Monetary'], 4, labels=[1, 2, 3, 4])
rfm['RFM_Score'] = rfm[['R_Score', 'F_Score', 'M_Score']].sum(axis=1)
st.write("RFM Analysis:\n", rfm.head())

### Model Prediction ###
if st.checkbox("Tampilkan Model Prediksi"):
    st.subheader("Model Prediksi: Linear Regression")

    # Define feature and target
    X = hour_df[['temp', 'hum', 'windspeed']]
    y = hour_df['cnt']
    
    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Predict
    y_pred = model.predict(X_test)
    
    # Calculate and display metrics
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    st.write(f"Mean Squared Error: {mse:.2f}")
    st.write(f"R^2 Score: {r2:.2f}")
