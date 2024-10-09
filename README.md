# Bike Sharing Analysis Dashboard 🚴‍♂️

## Overview
This project provides an analysis and visualization of the Bike Sharing Dataset. The dashboard allows users to explore rental patterns, seasonal trends, and perform RFM analysis to understand customer behavior better.

## Features
- Display information and descriptive statistics of the datasets.
- Visualizations of bike rentals by hour, season, and day of the week.
- Heatmap of correlations between numerical features.
- RFM (Recency, Frequency, Monetary) analysis for understanding customer segments.
- Predictive model to estimate bike rentals based on temperature, humidity, and wind speed.

## Datasets
The analysis uses the following datasets:
- `hour.csv`: Contains hourly rental data with features such as temperature, humidity, and wind speed.
- `day.csv`: Contains daily rental data with aggregated counts of rentals.

## Setup Environment - Anaconda
1. Create a new conda environment:
   ```bash
   conda create --name bike-sharing-analysis python=3.9.18
2. conda activate bike-sharing-analysis
3. pip install -r requirements.txt

## Setup Environment - Shell/Terminal
mkdir bike_sharing_analysis
cd bike_sharing_analysis
pipenv install
pipenv shell
pip install -r requirements.txt

## Run Streamlit
streamlit run Dashboard/Bike_Sharing_Analysis.py

