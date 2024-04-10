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

# Load the California housing dataset
california_housing = fetch_california_housing(as_frame=True)
X = california_housing.data
Y = california_housing.target

# Assuming we have longitude and latitude to approximately group by district
# Generate some mock district labels for the sake of example
# In a real-world scenario, you would use actual geographic division data such as zip codes
X['district'] = pd.cut(X['Longitude'], bins=10, labels=[f"District {i}" for i in range(10)])

# Calculate median house values by district
district_values = X.groupby('district').apply(
    lambda df: pd.Series({
        'MedianHouseValue': Y.loc[df.index].median()
    })
).reset_index().sort_values('MedianHouseValue', ascending=False)

# Sidebar for user inputs
with st.sidebar:
    st.title('🏠 California Housing Dashboard')
    
    # Median Income input
    median_income = st.slider('Median Income', float(X['MedInc'].min()), float(X['MedInc'].max()), float(X['MedInc'].median()))
    
    # House Age input
    house_age = st.slider('House Age', int(X['HouseAge'].min()), int(X['HouseAge'].max()), int(X['HouseAge'].median()))
    
    # Prepare user input features for prediction
    user_input = pd.DataFrame({'MedInc': [median_income], 'HouseAge': [house_age]})



# Update the model and predictions when the user input changes
def update_predictions(median_income, house_age):
    # Filter the data based on user input
    user_input = X.copy()
    user_input['MedInc'] = median_income
    user_input['HouseAge'] = house_age
    
    # Fit the model and predict values based on the user input
    model = DecisionTreeRegressor()
    model.fit(user_input[['MedInc', 'HouseAge']], Y)
    predictions = model.predict(user_input[['MedInc', 'HouseAge']])
    return predictions


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
    
    # Get the updated predictions
    X['PredictedValue'] = update_predictions(median_income, house_age)

    # Create a scatter plot
    plt.figure(figsize=(10, 6))
    plt.scatter(X['Longitude'], X['Latitude'], c=X['PredictedValue'], cmap='viridis', s=X['PredictedValue'])
    plt.colorbar(label='Predicted Median House Value')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title('Geographical Distribution of Median House Value')
    st.pyplot(plt)


# Column 3: Feature importances
with col3:
    st.markdown('#### Top Districts by Median House Value')
    
    # Format the district_values DataFrame for display
    district_values_display = district_values.copy()
    district_values_display['MedianHouseValue'] = district_values_display['MedianHouseValue'].apply(lambda x: f"${x:,.0f}")
    
    # Find max median house value for the progress bars
    max_value = district_values['MedianHouseValue'].max()
    
    # Create a custom column config with progress bars for the median house value
    district_values_display['Progress'] = district_values['MedianHouseValue'].apply(lambda x: x / max_value)
    
    # Display the dataframe with progress bars
    st.dataframe(district_values_display,
                 column_order=('District', 'MedianHouseValue', 'Progress'),
                 hide_index=True,
                 width=None,
                 column_config={
                     'district': st.column_config.TextColumn('district'),
                     'MedianHouseValue': st.column_config.TextColumn('Median House Value'),
                     'Progress': st.column_config.ProgressColumn('Median House Value', format="%f")
                 })
    
    

st.write("---")
# Additional analysis and visualizations can be added below

# Display the full map with actual house values for comparison
st.subheader('Actual Median House Value Map')
fig_full_map = px.scatter_mapbox(
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
st.plotly_chart(fig_full_map, use_container_width=True)

st.write("---")

'''
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
st.plotly_chart(fig, use_container_width=True)'''
