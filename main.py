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
    zoom=3,
    hover_name="district",  # Add a hover name for the district
    hover_data={"median_house_value": True},  # Add hover data for median house value
)

# Add the predicted house location with a distinct color and legend
fig.add_trace(px.scatter_mapbox(
    predicted_house, 
    lat="latitude", 
    lon="longitude", 
    color="PredictedValue", 
    size="PredictedValue", 
    color_continuous_scale=[[0, 'red'], [1, 'red']], 
    size_max=15,
    name="Predicted House"  # Add a name for the predicted house legend
).data[0])

# Add a legend for the median house value
fig.update_layout(
    mapbox=dict(
        layers=[
            dict(
                sourcetype="geojson",
                source=median_house_value_data,  # Replace median_house_value_data with your GeoJSON data
                type="fill",
                color="rgba(0,0,0,0)",  # Make the color transparent
                below="water",
                # Add a legend entry for median house value
                legend=dict(
                    title="Median House Value",
                    itemsizing="constant"
                ),
            ),
        ]
    )
)

# Add a pointer with an arrow
fig.add_annotation(
    x=longitude,  # X-coordinate of the pointer
    y=latitude,  # Y-coordinate of the pointer
    ax=longitude + 0.05,  # X-coordinate of the arrow
    ay=latitude + 0.05,  # Y-coordinate of the arrow
    axref="longitude",
    ayref="latitude",
    xref="longitude",
    yref="latitude",
    showarrow=True,  # Show the arrow
    arrowhead=2,  # Set the arrowhead style
    arrowsize=1,  # Set the arrow size
    arrowwidth=2,  # Set the arrow width
    arrowcolor="black",  # Set the arrow color
    text="Predicted House",  # Add text to the pointer
    font=dict(
        family="Arial",
        size=12,
        color="black"
    ),
)

st.plotly_chart(fig, use_container_width=True)

st.write("---")

st.markdown('#### Top Districts by Median House Value')
# You can add code here to display top districts by median house value
# Create a fictitious 'district' column
housing['district'] = ['District A', 'District B', 'District C', 'District A', 'District B'] * (len(housing) // 5)

# Group by the fictitious 'district' column and calculate the median house value
top_districts = housing.groupby('district')['median_house_value'].median().nlargest(5)

# Display the top districts
st.write(top_districts)

