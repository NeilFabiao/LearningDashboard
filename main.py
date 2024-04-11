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

# Define columns layout for the main panel
col1, col2, col3 = st.columns((2, 3, 2), gap="medium")

# Column 1: Input parameters and prediction
with col1:
    st.markdown('### Prediction')

    # Predict the price
    prediction = model.predict(user_input_df)[0]
    st.metric(label="Predicted Median House Value", value=f"${prediction * 1000:,.0f}")

# Column 2: Geographical distribution of median house value
with col2:
    st.markdown('### Geographical Distribution of Median House Value')
    
    # Create a scatter plot
    fig = px.scatter_mapbox(
        housing, 
        lat="latitude", 
        lon="longitude", 
        color="median_house_value", 
        size="median_house_value", 
        color_continuous_scale='viridis', 
        size_max=15, 
        zoom=5
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
