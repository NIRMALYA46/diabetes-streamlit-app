import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
st.set_page_config(page_title="Diabetes ML Streamlit App", layout="wide")
st.title("🩺 Diabetes Prediction ML App")
st.markdown(
    "Upload a diabetes dataset, explore the data, visualize patterns, and build a machine learning classifier."
)
st.sidebar.header("Data Options")
upload = st.sidebar.file_uploader("Upload your diabetes CSV file", type=["csv"])
if upload:
    df = pd.read_csv(upload)
else:
    df = pd.read_csv(r"C:\Users\XPS\Downloads\diabetes.csv")

st.write("**Dataset Preview:**")
st.write(df.head())

# EXPLORATORY DATA ANALYSIS
st.subheader("Exploratory Data Analysis")
st.write("**Summary statistics:**")
st.write(df.describe())
if st.checkbox("Show Correlation Heatmap"):
    fig, ax = plt.subplots()
    sns.heatmap(df.corr(), annot=True, cmap="YlGnBu", ax=ax)
    st.pyplot(fig)

if st.checkbox("Show Histogram Plot"):
    feature = st.selectbox("Choose feature for histogram", df.columns[:-1])
    fig, ax = plt.subplots()
    sns.histplot(df[feature], kde=True, ax=ax)
    st.pyplot(fig)

st.subheader("Outcome Distribution")
fig, ax = plt.subplots()
sns.countplot(x="Outcome", data=df, ax=ax)
st.pyplot(fig)

st.sidebar.header("ML Model Options")
features = st.sidebar.multiselect(
    "Select features", list(df.columns[:-1]), default=list(df.columns[:-1])
)
target = "Outcome"

X = df[features]
y = df[target]

test_size = st.sidebar.slider("Test set size", 0.1, 0.5, 0.2, step=0.05)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

params = st.sidebar.expander("Model Parameters", expanded=False)
n_estimators = params.slider("n_estimators", 10, 200, 100, step=10)
max_depth = params.slider("max_depth", 2, 20, 5, step=1)

rf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
rf.fit(X_train, y_train)
preds = rf.predict(X_test)
acc = accuracy_score(y_test, preds)

st.markdown(f"**🔍 Model Accuracy:** {acc:.2%}")

# CLASSIFICATION REPORT
if st.checkbox("Show Classification Report"):
    st.text(classification_report(y_test, preds))

# CONFUSION MATRIX
if st.checkbox("Show Confusion Matrix"):
    cm = confusion_matrix(y_test, preds)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)

# PREDICTION ON CUSTOM INPUT
st.subheader("Predict Diabetes for Custom Input")
input_data = []
for col in features:
    val = st.number_input(
        f"Input {col}:",
        float(df[col].min()),
        float(df[col].max()),
        float(df[col].mean())
    )
    input_data.append(val)
if st.button("Predict Diabetes"):
    pred = rf.predict([input_data])[0]
    label = "Diabetic (1)" if pred == 1 else "Not Diabetic (0)"
    st.success(f"Prediction: {label}")