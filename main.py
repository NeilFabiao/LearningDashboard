import streamlit as st
import pandas as pd
import shap
from sklearn.datasets import fetch_california_housing
from sklearn.tree import DecisionTreeRegressor 
#from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import plotly.express as px  

st.write("""
# California House Price Prediction App

This app predicts the **California House Price**!
""")
st.write('---')

# Loads the California Housing Dataset
california_housing = fetch_california_housing(as_frame=True)
X = california_housing.data
Y = california_housing.target
X['MedHouseValue'] = Y

# Sidebar - Specify Input Parameters
st.sidebar.header('Specify Input Parameters')

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
            'Longitude': Longitude}
    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()

# Main Panel

# Print specified input parameters
st.header('Specified Input parameters')
st.write(df)
st.write('---')


# Build Regression Model
model = DecisionTreeRegressor()
model.fit(X, Y)
# Apply Model to Make Prediction
prediction = model.predict(df)


st.header('Prediction of Median House Value')
predicted_value = float(prediction[0])
st.write(f"The median house value is : ${predicted_value * 100000:,.2f}")
st.write('---')

st.header('Map Visualization')

# Map Visualization
st.header('Geographical Distribution of Data')
fig = px.scatter_mapbox(
    X,
    lat="Latitude",
    lon="Longitude",
    size="MedInc",
    color="MedHouseValue",
    color_continuous_scale=px.colors.cyclical.IceFire,
    size_max=15,
    zoom=5,
    mapbox_style="carto-positron"
)
st.plotly_chart(fig, use_container_width=True)

st.write('---')

st.header('Feature Importance')
plt.title('Feature importance based on SHAP values')

# Explaining the model's predictions using SHAP values
explainer = shap.Explainer(model)  # shap.TreeExplainer is now just shap.Explainer in newer versions of SHAP
shap_values = explainer.shap_values(X)

shap.summary_plot(shap_values, X)
st.pyplot(bbox_inches='tight')
st.write('---')


plt.title('Feature importance based on SHAP values (Bar)')
shap.summary_plot(shap_values, X, plot_type="bar")
st.pyplot(bbox_inches='tight')



