import streamlit as st
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder

# Load the Iris dataset
iris = load_iris()

# Create a DataFrame
iris_df = pd.DataFrame(iris.data, columns=iris.feature_names)

# Add species names
iris_df['species'] = [iris.target_names[i] for i in iris.target]

# Encode species to numeric values
label_encoder = LabelEncoder()
iris_df['encoded_species'] = label_encoder.fit_transform(iris_df['species'])

# Model selection
model_type = st.sidebar.selectbox(
    "Select Model",
    ("Linear Regression", "Logistic Regression", "SVM")
)
# Streamlit app title
st.title("Iris Flower Classification")

# Sidebar for input
st.sidebar.header("Input Flower Measurements")
sepal_length = st.sidebar.number_input("Sepal Length (cm)", min_value=0.1, max_value=8.0, value=5.1)
sepal_width = st.sidebar.number_input("Sepal Width (cm)", min_value=0.1, max_value=4.5, value=3.5)
petal_length = st.sidebar.number_input("Petal Length (cm)", min_value=0.1, max_value=7.0, value=1.4)
petal_width = st.sidebar.number_input("Petal Width (cm)", min_value=0.1, max_value=2.5, value=0.2)



# Prepare features and target
X = iris_df[['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']]
y = iris_df['encoded_species']

# Train models
if model_type == "Linear Regression":
    model = LinearRegression()
elif model_type == "Logistic Regression":
    model = LogisticRegression(max_iter=200)
else:  # SVM
    model = SVC(probability=True)

model.fit(X, y)

# Make prediction from user input
input_values = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

if model_type == "Linear Regression":
    predicted_value = model.predict(input_values)[0]
    predicted_class = int(round(predicted_value))
else:  # For classification models
    predicted_class = model.predict(input_values)[0]
    predicted_value = model.predict_proba(input_values)[0] if hasattr(model, 'predict_proba') else [0, 0, 0]

# Convert predicted class to species name
species_mapping = {
    0: 'Setosa',
    1: 'Versicolor',
    2: 'Virginica'
}
predicted_species = species_mapping.get(predicted_class, "Unknown")

# Display results
st.subheader("Results")
st.write(f"Using model: **{model_type}**")
st.write("Based on the values you entered:")

st.success(f"Predicted Species: {predicted_species}")

if model_type == "Linear Regression":
    st.write(f"**Regression Output Value:** {predicted_value:.2f}")
    st.write(f"**Rounded Prediction (Class):** {predicted_class}")
else:
    if hasattr(model, 'predict_proba'):
        st.write("**Class Probabilities:**")
        for i, prob in enumerate(predicted_value):
            st.write(f"{species_mapping[i]}: {prob:.2f}")