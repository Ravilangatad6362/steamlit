import streamlit as st
import numpy as np
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler

# Load and preprocess the data
iris = load_iris()
scaler = StandardScaler()
X = scaler.fit_transform(iris.data)
model = DecisionTreeClassifier(random_state=42).fit(X, iris.target)

# Streamlit app
st.title("Iris Flower Prediction App")
st.write("Enter the flower measurements below:")

# Input sliders
sepal_length = st.slider("Sepal Length", float(iris.data[:, 0].min()), float(iris.data[:, 0].max()))
sepal_width = st.slider("Sepal Width", float(iris.data[:, 1].min()), float(iris.data[:, 1].max()))
petal_length = st.slider("Petal Length", float(iris.data[:, 2].min()), float(iris.data[:, 2].max()))
petal_width = st.slider("Petal Width", float(iris.data[:, 3].min()), float(iris.data[:, 3].max()))

# Predict button
if st.button("Predict"):
    input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    input_data_scaled = scaler.transform(input_data)
    prediction = model.predict(input_data_scaled)
    predicted_class = iris.target_names[prediction[0]]
    st.success(f"The predicted class is: {predicted_class}")
