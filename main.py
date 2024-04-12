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

# Column 2: Geographical distribution of predicted house
with col2:
    st.markdown('### Geographical Distribution of Predicted House')

    # Predict the price for the user input
    prediction = model.predict(user_input_df)[0]

    # Find the nearest record in the dataset to the predicted price
    nearest_record = housing.iloc[(housing['median_house_value'] - prediction).abs().argsort()[:1]]

    # Extract latitude and longitude from the nearest record
    latitude = nearest_record['latitude'].values[0]
    longitude = nearest_record['longitude'].values[0]

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
    The data was originally featured in a paper titled "Sparse spatial autoregressions" by 
    R. Kelley Pace and Ronald Barry. This dataset is a modified version of the California Housing dataset, 
    which is available from Luís Torgo's page at the University of Porto. 
    The information was encountered in "Hands-On Machine learning with Scikit-Learn and TensorFlow" by 
    [Aurélien Géron](https://inria.github.io/scikit-learn-mooc/python_scripts/datasets_california_housing.html).

    """)

# Additional analysis and visualizations can be added below

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
    zoom=3
)
fig.update_layout(mapbox_style="carto-positron")

# Add the predicted house location with a distinct color and legend
predicted_trace = px.scatter_mapbox(
    predicted_house, 
    lat="latitude", 
    lon="longitude", 
    color="PredictedValue", 
    size="PredictedValue", 
    color_continuous_scale=[[0, 'red'], [1, 'red']], 
    size_max=15
).data[0]

fig.add_trace(predicted_trace)

# Add annotation for the predicted house
fig.add_annotation(
    x=predicted_house['longitude'].values[0],  # X coordinate of the arrow tail
    y=predicted_house['latitude'].values[0],   # Y coordinate of the arrow tail
    xref="x",
    yref="y",
    ax=0,  # X component of the arrowhead
    ay=-40,  # Y component of the arrowhead
    arrowhead=2,
    arrowsize=1.5,
    arrowwidth=1,
    arrowcolor="black",
    opacity=0.7,
    text="Predicted House",  # Annotation text
    font=dict(color="black", size=12),
    showarrow=True,
    arrowhead=3,
    arrowwidth=1,
    arrowcolor="black"
)

st.write("---")

st.markdown('#### Top Districts by Median House Value')
# You can add code here to display top districts by median house value
# Create a fictitious 'district' column
housing['district'] = ['District A', 'District B', 'District C', 'District A', 'District B'] * (len(housing) // 5)

# Group by the fictitious 'district' column and calculate the median house value
top_districts = housing.groupby('district')['median_house_value'].median().nlargest(5)

# Display the top districts
st.write(top_districts)

