import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.datasets import fetch_california_housing
from sklearn.tree import DecisionTreeRegressor

# Set up the page configuration
st.set_page_config(
    page_title="California Housing Dashboard",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

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

# Define columns layout for the main panel
col1, col2, col3 = st.columns((2, 3, 2), gap="medium")

# Column 1: Input parameters and prediction
with col1:
    st.markdown('### Prediction')
    
    # Build and fit the model
    model = DecisionTreeRegressor()
    model.fit(X[['MedInc', 'HouseAge']], Y)
    
    # Predict the price
    prediction = model.predict(user_input)[0]
    st.metric(label="Predicted Median House Value", value=f"${prediction * 1000:,.0f}")

# Column 2: Geographical distribution of median house value
with col2:
    st.markdown('### Geographical Distribution')
    
    # Create a map visualization
    fig = px.scatter_mapbox(
        california_housing.frame,
        lat="Latitude",
        lon="Longitude",
        size="MedHouseVal",
        color="MedHouseVal",
        color_continuous_scale=px.colors.cyclical.IceFire,
        size_max=15,
        zoom=5,
        mapbox_style="carto-positron"
    )
    st.plotly_chart(fig, use_container_width=True)

# Column 3: Feature importances
with col3:
    st.markdown('### Feature Importance')
    
    # Re-fit model with all features for feature importance
    model_all_features = DecisionTreeRegressor()
    model_all_features.fit(X, Y)
    
    # Plot feature importance
    features = pd.DataFrame({
        'Feature': X.columns,
        'Importance': model_all_features.feature_importances_
    }).sort_values(by='Importance', ascending=False)

    fig = px.bar(
        features,
        x='Importance',
        y='Feature',
        orientation='h',
        color='Importance',
        title='Feature Importances in Predicting House Prices'
    )
    st.plotly_chart(fig, use_container_width=True)

# Additional analysis and visualizations can be added below
