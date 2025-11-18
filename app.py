import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris

st.set_page_config(page_title="Iris Classifier", page_icon="🌸", layout="wide")

st.title("🌸 Iris Classifier Dashboard")
st.write("Upload a CSV or use the default Iris dataset to predict *Setosa*, *Versicolor*, *Virginica*.")

# ----------------------------
# Sidebar
# ----------------------------
st.sidebar.header("📁 Upload Your Dataset")
uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=["csv"])

st.sidebar.header("🔧 Input Features")

sepal_length = st.sidebar.number_input("Sepal Length (cm)", 0.0, 10.0, 5.8)
sepal_width = st.sidebar.number_input("Sepal Width (cm)", 0.0, 10.0, 3.0)
petal_length = st.sidebar.number_input("Petal Length (cm)", 0.0, 10.0, 4.35)
petal_width = st.sidebar.number_input("Petal Width (cm)", 0.0, 10.0, 1.3)

user_input = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

# ----------------------------
# Load Data
# ----------------------------
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.success("CSV loaded successfully!")

    st.subheader("📁 Uploaded Dataset Preview")
    st.dataframe(df.head())

    # Auto-detect target column
    if "species" in df.columns:
        target_col = "species"
    elif "target" in df.columns:
        target_col = "target"
    else:
        st.error("❌ No 'species' or 'target' column found in uploaded CSV.")
        st.stop()

    X = df.drop(target_col, axis=1)
    y = df[target_col]

    # Encode species if needed
    if y.dtype == object:
        y = y.astype("category").cat.codes

    species_map = {0: "Setosa", 1: "Versicolor", 2: "Virginica"}

else:
    st.info("No CSV uploaded — using default Iris dataset.")
    iris = load_iris()
    X = pd.DataFrame(iris.data, columns=iris.feature_names)
    y = iris.target
    species_map = {0: "Setosa", 1: "Versicolor", 2: "Virginica"}

    df = X.copy()
    df["species"] = pd.Series(y).map(species_map)

# ----------------------------
# Train Model
# ----------------------------
model = RandomForestClassifier(n_estimators=120, random_state=42)
model.fit(X, y)

# ----------------------------
# Prediction
# ----------------------------
prediction = model.predict(user_input)[0]
predicted_label = species_map[prediction]

st.markdown("---")
st.subheader("🌼 Prediction Result")
st.success(f"Predicted Species: **{predicted_label}**")

st.subheader("📊 Input Summary")
st.table(pd.DataFrame(user_input, columns=X.columns[:4]))

st.markdown("---")

st.subheader("📈 Dataset Statistics")
st.write(df.describe())
