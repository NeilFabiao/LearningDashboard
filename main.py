import streamlit as st
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt
import plotly.express as px

st.write("""
# California House Price Prediction App

This app predicts the **California House Price**!
""")
st.write('---')

# Load California housing dataset
california_housing = fetch_california_housing(as_frame=True)
X = california_housing.data
Y = california_housing.target

# Sidebar - Specify Input Parameters
st.sidebar.header('House Price Prediction')
st.sidebar.subheader('Specify Input Parameters')

def user_input_features():
    MedInc = st.sidebar.slider('Median Income', X['MedInc'].min(), X['MedInc'].max(), X['MedInc'].mean())
    HouseAge = st.sidebar.slider('House Age', X['HouseAge'].min(), X['HouseAge'].max(), X['HouseAge'].mean())
    AveRooms = st.sidebar.slider('Average Rooms', X['AveRooms'].min(), X['AveRooms'].max(), X['AveRooms'].mean())
    AveBedrms = st.sidebar.slider('Average Bedrooms', X['AveBedrms'].min(), X['AveBedrms'].max(), X['AveBedrms'].mean())
    Population = st.sidebar.slider('Population', X['Population'].min(), X['Population'].max(), X['Population'].mean())
    AveOccup = st.sidebar.slider('Average Occupancy', X['AveOccup'].min(), X['AveOccup'].max(), X['AveOccup'].mean())
    Latitude = st.sidebar.slider('Latitude', X['Latitude'].min(), X['Latitude'].max(), X['Latitude'].mean())
    Longitude = st.sidebar.slider('Longitude', X['Longitude'].min(), X['Longitude'].max(), X['Longitude'].mean())
    data = {'MedInc': MedInc,
            'HouseAge': HouseAge,
            'AveRooms': AveRooms,
            'AveBedrms': AveBedrms,
            'Population': Population,
            'AveOccup': AveOccup,
            'Latitude': Latitude,
            'Longitude': Longitude
           }
    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()

# Main Panel

# Build Regression Model
model = DecisionTreeRegressor()
model.fit(X, Y)

# Apply Model to Make Prediction
prediction = model.predict(df)

# Display Predicted House Price and Map Visualization
st.subheader('Prediction and Map Visualization of Median House Value ')
predicted_value = float(prediction[0])
st.write(f"The median house value is : ${predicted_value * 100000:,.2f}")
st.write('---')

st.subheader('Geographical Distribution of Predicted House Prices')
fig = px.scatter_mapbox(
    df,
    lat="Latitude",
    lon="Longitude",
    size=[predicted_value],  # Size based on predicted house price
    color=[predicted_value],  # Color based on predicted house price
    color_continuous_scale=px.colors.cyclical.IceFire,
    size_max=15,
    zoom=5,
    mapbox_style="carto-positron"
)
st.plotly_chart(fig, use_container_width=True)
st.write('---')

# Feature Importance Visualization
st.subheader('Feature Importance based on Decision Tree')

# Bar plot of feature importance
feature_importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': model.feature_importances_})
fig = px.bar(feature_importance_df.sort_values(by='Importance', ascending=False), x='Feature', y='Importance',
             labels={'Importance': 'Feature Importance'}, color='Importance')
st.plotly_chart(fig, use_container_width=True)
st.write('---')

# Analysis of Housing Prices Across Different Regions
st.subheader('Analysis of Housing Prices Across Different Regions')

# Scatter plot of Latitude vs. Longitude colored by house prices
fig = px.scatter_mapbox(
    pd.concat([X[['Latitude', 'Longitude']], pd.DataFrame({'HousePrice': Y})], axis=1),
    lat="Latitude",
    lon="Longitude",
    color="HousePrice",
    color_continuous_scale="Viridis",
    zoom=5,
    mapbox_style="carto-positron"
)
st.plotly_chart(fig, use_container_width=True)
st.write('---')

# Relationship Between Median Income and Housing Prices
st.subheader('Relationship Between Median Income and Housing Prices')

# Scatter plot of Median Income vs. House Price
fig = px.scatter(
    pd.concat([X[['MedInc']], pd.DataFrame({'HousePrice': Y})], axis=1),
    x='MedInc',
    y='HousePrice',
    trendline='ols',
    labels={'HousePrice': 'Median House Price ($)', 'MedInc': 'Median Income'},
)
st.plotly_chart(fig, use_container_width=True)
st.write('---')
