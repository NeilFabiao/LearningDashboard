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
california_housing = fetch_california_housing(as_frame=True)
X = california_housing.data
Y = california_housing.target

# Sidebar for user inputs
with st.sidebar:
    st.title('🏠 California Housing Dashboard')
    
    # Median Income input
    median_income = st.slider('Median Income', float(X['MedInc'].min()), float(X['MedInc'].max()), float(X['MedInc'].median()))
    
    # House Age input
    house_age = st.slider('House Age', int(X['HouseAge'].min()), int(X['HouseAge'].max()), int(X['HouseAge'].median()))
    
    # Prepare user input features for prediction
    user_input = pd.DataFrame({'MedInc': [median_income], 'HouseAge': [house_age]})



    
    

