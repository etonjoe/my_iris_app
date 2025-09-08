# app.py

import streamlit as st
import pandas as pd
import joblib

# --- Page Configuration ---
st.set_page_config(
    page_title="Iris Species Classifier",
    page_icon="🌸",
    layout="centered"
)

# --- Load The Model and Species Names ---
# We use a try-except block to handle potential FileNotFoundError
try:
    model = joblib.load('iris_classifier_model.joblib')
    species_names = joblib.load('iris_species_names.joblib')
except FileNotFoundError:
    st.error("Model or species file not found. Please ensure they are in the correct directory.")
    st.stop() # Stop the app from running further if files are missing

# --- App Title and Description ---
st.title("🌸 Iris Flower Species Classifier")
st.markdown("""
This app uses a simple Logistic Regression model to predict the species of an Iris flower 
based on its sepal and petal measurements.
""")

# --- Sidebar for User Input ---
st.sidebar.header("Input Flower Measurements")

# Create sliders in the sidebar for user input
sepal_length = st.sidebar.slider("Sepal Length (cm)", min_value=4.0, max_value=8.0, value=5.4, step=0.1)
sepal_width = st.sidebar.slider("Sepal Width (cm)", min_value=2.0, max_value=4.5, value=3.4, step=0.1)
petal_length = st.sidebar.slider("Petal Length (cm)", min_value=1.0, max_value=7.0, value=1.6, step=0.1)
petal_width = st.sidebar.slider("Petal Width (cm)", min_value=0.1, max_value=2.5, value=0.2, step=0.1)

# --- Prediction Logic ---
# Create a button to trigger the prediction
if st.sidebar.button("Predict Species", type="primary"):

    # Create a DataFrame from the user's input
    # The column names MUST match the names used during model training
    input_data = pd.DataFrame({
        'sepal length (cm)': [sepal_length],
        'sepal width (cm)': [sepal_width],
        'petal length (cm)': [petal_length],
        'petal width (cm)': [petal_width]
    })
    
    # Make a prediction
    prediction_index = model.predict(input_data)[0]
    prediction_name = species_names[prediction_index]
    
    # Get prediction probabilities
    prediction_proba = model.predict_proba(input_data)[0]
    
    # --- Display the Result ---
    st.subheader(f"Prediction: **{prediction_name.title()}**")
    
    confidence = prediction_proba[prediction_index] * 100
    st.write(f"Confidence: **{confidence:.2f}%**")

    # Display probabilities in a more visual way
    st.write("Prediction Probabilities:")
    st.bar_chart(pd.DataFrame(prediction_proba.reshape(1, -1), columns=species_names))

else:
    st.info("Adjust the sliders in the sidebar and click 'Predict Species' to see the result.")