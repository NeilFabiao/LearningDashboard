import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt

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

# Streamlit UI setup
st.set_page_config(page_title="California Housing Dashboard", page_icon="🏠", layout="wide",initial_sidebar_state="expanded")
st.title("Housing Price Prediction App 🏠")
st.write("This app predicts the median house value based on the California House Price data.")

st.write('---')

# Sidebar for user inputs
with st.sidebar:
    st.title("Input Features")
    median_income = st.number_input("Median Income", value=float(X_train['MedInc'].median()))
    house_age = st.number_input("House Age", value=float(X_train['HouseAge'].median()))

# Function to update predictions based on user input
def update_predictions(median_income, house_age):
    user_input = X_train.copy()
    user_input['MedInc'] = median_income
    user_input['HouseAge'] = house_age
    
    predictions = model.predict(user_input)
    return predictions
