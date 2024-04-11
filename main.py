import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.datasets import fetch_california_housing
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt

# Set up the page configuration
st.set_page_config(
    page_title="California Housing Dashboard",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.write("""
# California House Price Prediction App 🏠

This app predicts the **California House Price**!
""")
st.write('---')

# Load the California housing dataset
california_housing = pd.read_csv('housing.csv')

# 
X = california_housing.drop('median_house_value', axis=1)  # Features (drop the target column)
Y = california_housing['median_house_value']  # Target variable (assuming 'Target' is the column name)


# Sidebar for user inputs
with st.sidebar:
    st.title('🏠 California Housing Dashboard')
    
    # Median Income input
    median_income = st.slider('Median Income', float(X['median_income'].min()), float(X['median_income'].max()), float(X['median_income'].median()))
    
    # House Age input
    house_age = st.slider('House Age', int(X['housing_median_age'].min()), int(X['housing_median_age'].max()), int(X['housing_median_age'].median()))
    
    # Prepare user input features for prediction
    user_input = pd.DataFrame({'MedInc': [median_income], 'HouseAge': [house_age]})



    
    

