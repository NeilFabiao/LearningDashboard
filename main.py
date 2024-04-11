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
        'longitude': st.slider('Longitude', housing['longitude'].min(), housing['longitude'].max(), housing['longitude'].median()),
        'latitude': st.slider('Latitude', housing['latitude'].min(), housing['latitude'].max(), housing['latitude'].median()),
        'housing_median_age': st.slider('Housing Median Age', housing['housing_median_age'].min(), housing['housing_median_age'].max(), housing['housing_median_age'].median()),
        'total_rooms': st.slider('Total Rooms', housing['total_rooms'].min(), housing['total_rooms'].max(), housing['total_rooms'].median()),
        'total_bedrooms': st.slider('Total Bedrooms', housing['total_bedrooms'].min(), housing['total_bedrooms'].max(), housing['total_bedrooms'].median()),
        'population': st.slider('Population', housing['population'].min(), housing['population'].max(), housing['population'].median()),
        'households': st.slider('Households', housing['households'].min(), housing['households'].max(), housing['households'].median()),
        'median_income': st.slider('Median Income', housing['median_income'].min(), housing['median_income'].max(), housing['median_income'].median()),
        'ocean_proximity': st.selectbox('Ocean Proximity', ocean_proximity_options)
    }

# Convert user input to DataFrame
user_input_df = pd.DataFrame([user_input])

# One-hot encode the 'ocean_proximity' column
encoder = OneHotEncoder()
ocean_proximity_encoded = encoder.fit_transform(user_input_df[['ocean_proximity']]).toarray()
ocean_proximity_encoded_df = pd.DataFrame(ocean_proximity_encoded, columns=encoder.get_feature_names_out(['ocean_proximity']))

# Drop the original 'ocean_proximity' column and concatenate the one-hot encoded columns
user_input_df = user_input_df.drop('ocean_proximity', axis=1)
user_input_df = pd.concat([user_input_df, ocean_proximity_encoded_df], axis=1)

# Display user input
st.write("Your Input Features")
st.dataframe(user_input_df)

# Train the model with the entire dataset
X = housing.drop('median_house_value', axis=1)
Y = housing['median_house_value']
model = DecisionTreeRegressor()
model.fit(X, Y)

# Predict the price
prediction = model.predict(user_input_df)[0]

# Display prediction
st.write('---')
st.subheader('Prediction')
st.write(f"Predicted Median House Value: ${prediction * 1000:,.0f}")
