import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder

# Load the housing dataset from a CSV file
housing = pd.read_csv('housing.csv')

# Drop NaN values and duplicates
housing.dropna(inplace=True)
housing.drop_duplicates(inplace=True)

# One-hot encode the 'ocean_proximity' categorical variable
encoder = OneHotEncoder()
ocean_proximity_encoded = encoder.fit_transform(housing[['ocean_proximity']]).toarray()
ocean_proximity_encoded_df = pd.DataFrame(ocean_proximity_encoded, columns=encoder.get_feature_names_out(['ocean_proximity']))

# Drop the original 'ocean_proximity' column and concatenate the one-hot encoded columns
housing = housing.drop('ocean_proximity', axis=1).reset_index(drop=True)
housing = pd.concat([housing, ocean_proximity_encoded_df], axis=1)

# Split the dataset into features and target variable
X = housing.drop('median_house_value', axis=1)
Y = housing['median_house_value']

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Train the model with the training set
model = DecisionTreeRegressor()
model.fit(X_train, y_train)

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
        'longitude': st.slider('Longitude', float(X_train['longitude'].min()), float(X_train['longitude'].max()), float(X_train['longitude'].median())),
        'latitude': st.slider('Latitude', float(X_train['latitude'].min()), float(X_train['latitude'].max()), float(X_train['latitude'].median())),
        'housing_median_age': st.slider('Housing Median Age', int(X_train['housing_median_age'].min()), int(X_train['housing_median_age'].max()), int(X_train['housing_median_age'].median())),
        'total_rooms': st.slider('Total Rooms', int(X_train['total_rooms'].min()), int(X_train['total_rooms'].max()), int(X_train['total_rooms'].median())),
        'total_bedrooms': st.slider('Total Bedrooms', int(X_train['total_bedrooms'].min()), int(X_train['total_bedrooms'].max()), int(X_train['total_bedrooms'].median())),
        'population': st.slider('Population', int(X_train['population'].min()), int(X_train['population'].max()), int(X_train['population'].median())),
        'households': st.slider('Households', int(X_train['households'].min()), int(X_train['households'].max()), int(X_train['households'].median())),
        'median_income': st.slider('Median Income', float(X_train['median_income'].min()), float(X_train['median_income'].max()), float(X_train['median_income'].median())),
        'ocean_proximity': st.selectbox('Ocean Proximity', ocean_proximity_options)
    }

# Convert user input to DataFrame
user_input_df = pd.DataFrame([user_input])

# Display user input
st.write("Your Input Features")
st.dataframe(user_input_df)

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
        X, 
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
