# Import necessary libraries
import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.datasets import fetch_california_housing

# Load the housing dataset from sklearn
data = fetch_california_housing()
housing = pd.DataFrame(data.data, columns=data.feature_names)
housing['median_house_value'] = data.target  # Renaming the target column

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
        'median_income': st.slider('Median Income', housing['MedInc'].min(), housing['MedInc'].max(), housing['MedInc'].median()),
        'housing_median_age': st.slider('Housing Median Age', housing['HouseAge'].min(), housing['HouseAge'].max(), housing['HouseAge'].median()),
    }

# Convert user input to DataFrame
user_input_df = pd.DataFrame([user_input])

# Train the model with the entire dataset
X = housing[['MedInc', 'HouseAge']]
Y = housing['median_house_value']
model = DecisionTreeRegressor()
model.fit(X, Y)

# Define columns layout for the main panel
col1, col2, col3 = st.columns((2, 3, 2), gap="medium")

# Column 1: Input parameters and prediction
with col1:
    st.markdown('### Prediction')

    # Predict the price
    prediction = model.predict(user_input_df)[0]
    st.metric(label="Predicted Median House Value", value=f"${prediction * 1000:,.0f}")

# Column 2: Geographical distribution of predicted house
with col2:
    st.markdown('### Geographical Distribution of Predicted House')

    # Predict the price for the user input
    prediction = model.predict(user_input_df)[0]

    # Find the nearest record in the dataset to the predicted price
    nearest_record = housing.iloc[(housing['median_house_value'] - prediction).abs().argsort()[:1]]

    # Extract latitude and longitude from the nearest record
    latitude = nearest_record['Latitude'].values[0]
    longitude = nearest_record['Longitude'].values[0]

    # Create a DataFrame with predicted house values using the latitude and longitude from the nearest record
    predicted_house = pd.DataFrame({
        'latitude': [latitude],
        'longitude': [longitude],
        'PredictedValue': [prediction]
    })

    # Create a scatter plot for the predicted house
    fig = px.scatter_mapbox(
        predicted_house, 
        lat="latitude", 
        lon="longitude", 
        color="PredictedValue", 
        size="PredictedValue", 
        color_continuous_scale='viridis', 
        size_max=15, 
        zoom=3
    )
    fig.update_layout(mapbox_style="carto-positron")
    st.plotly_chart(fig, use_container_width=True)


# Column 3: Information about the data and top districts
with col3:
    st.markdown('### Dataset Citation and Reference')
    st.write("""
    This data was initially featured in the following paper:
    Pace, R. Kelley, and Ronald Barry. "Sparse spatial autoregressions." Statistics & Probability Letters 33.3 (1997): 291-297.
    and I encountered it in 'Hands-On Machine learning with Scikit-Learn and TensorFlow' by Aurélien Géron.
    Aurélien Géron wrote:
    This dataset is a modified version of the California Housing dataset available from:
    Luís Torgo's page (University of Porto)
    """)
    
    st.markdown('#### Top Districts by Median House Value')
    # You can add code here to display top districts by median house value

# Additional analysis and visualizations can be added below

st.markdown('### Geographical Distribution of Median House Value')
    
# Create a scatter plot
fig = px.scatter_mapbox(
    housing, 
    lat="Latitude", 
    lon="Longitude", 
    color="median_house_value", 
    size="median_house_value", 
    color_continuous_scale='viridis', 
    size_max=15, 
    zoom=3
)
fig.update_layout(mapbox_style="carto-positron")

# Add the predicted house location with a distinct color and legend
fig.add_trace(px.scatter_mapbox(
    predicted_house, 
    lat="latitude", 
    lon="longitude", 
    color="PredictedValue", 
    size="PredictedValue", 
    color_continuous_scale=[[0, 'red'], [1, 'red']], 
    size_max=15
).data[0])

st.plotly_chart(fig, use_container_width=True)
