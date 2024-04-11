# Import necessary libraries
import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import OneHotEncoder

# Load the housing dataset from a CSV file
housing = pd.read_csv('housing.csv')

# Drop NaN values and duplicates
housing.dropna(inplace=True)
housing.drop_duplicates(inplace=True)

# Define ocean proximity options
ocean_proximity_options = ['<1H OCEAN', 'INLAND', 'NEAR OCEAN', 'NEAR BAY', 'ISLAND']

# Streamlit UI setup
st.set_page_config(page_title="California Housing Dashboard", page_icon="🏠", layout="wide", initial_sidebar_state="expanded")
st.title("Housing Price Prediction 🏠")

st.write('---')

# Sidebar for user inputs
with st.sidebar:
    st.title("Input Features")
    # Collect user input
    user_input = {
        'total_bedrooms': st.slider('Total Bedrooms', housing['total_bedrooms'].min(), housing['total_bedrooms'].max(), housing['total_bedrooms'].median()),
        'median_income': st.slider('Median Income', housing['median_income'].min(), housing['median_income'].max(), housing['median_income'].median()),
        'housing_median_age': st.slider('Housing Median Age', housing['housing_median_age'].min(), housing['housing_median_age'].max(), housing['housing_median_age'].median()),
    }

# Convert user input to DataFrame
user_input_df = pd.DataFrame([user_input])

# Display user input
st.write("Your Input Features")
st.dataframe(user_input_df)

# Train the model with the entire dataset
X = housing[['total_bedrooms', 'median_income', 'housing_median_age']]
Y = housing['median_house_value']
model = DecisionTreeRegressor()
model.fit(X, Y)

# Predict the price
prediction = model.predict(user_input_df)[0]

# Display prediction
st.write('---')
st.subheader('Prediction')
st.write(f"Predicted Median House Value: ${prediction * 1000:,.0f}")
