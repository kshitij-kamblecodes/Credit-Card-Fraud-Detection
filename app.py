import streamlit as st
import pandas as pd
import joblib

# ---------------- PAGE CONFIG ---------------- #
st.set_page_config(
    page_title="Credit Card Fraud Detection",
    layout="wide"
)

# ---------------- CUSTOM CSS ---------------- #
st.markdown("""
<style>

/* Gradient Background */
.stApp {
    background: linear-gradient(135deg, #e6f0fa 0%, #f8fbff 100%);
}

/* Center main content */
.main-container {
    max-width: 950px;
    margin: auto;
    padding-top: 50px;
}

/* Header styling */
.header {
    display: flex;
    align-items: center;
    gap: 20px;
    margin-bottom: 35px;
}

.header-title {
    font-size: 38px;
    font-weight: 700;
    color: #1f2937;
}

.header-subtitle {
    font-size: 17px;
    color: #4b5563;
}

/* Sidebar Styling */
section[data-testid="stSidebar"] {
    background-color: #f9fafb;
}

</style>
""", unsafe_allow_html=True)

# ---------------- SIDEBAR ---------------- #
st.sidebar.markdown("## ⚙ Model Selection")

model_choice = st.sidebar.selectbox(
    "Choose Model",
    ["Logistic Regression", "Random Forest"]
)

# ---------------- LOAD MODELS ---------------- #
@st.cache_resource
def load_models():
    lr = joblib.load("models/logistic_regression.pkl")
    rf = joblib.load("models/random_forest.pkl")
    return lr, rf

lr, rf = load_models()

# ---------------- MAIN CONTENT ---------------- #
st.markdown('<div class="main-container">', unsafe_allow_html=True)

# Header
st.markdown("""
<div class="header">
    <img src="https://cdn-icons-png.flaticon.com/512/633/633611.png" width="75">
    <div>
        <div class="header-title">Credit Card Fraud Detection</div>
        <div class="header-subtitle">
            Predict whether a transaction is <b>Fraudulent</b> or <b>Legitimate</b>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ✅ NO WHITE CARD HERE

st.markdown("### Upload transaction CSV file")

uploaded_file = st.file_uploader(
    "Drag and drop file here",
    type=["csv"]
)

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.markdown("### Uploaded Data Preview")
    st.dataframe(df.head(), use_container_width=True)

    if "Class" in df.columns:
        X = df.drop("Class", axis=1)
    else:
        X = df

    if model_choice == "Logistic Regression":
        predictions = lr.predict(X)
    else:
        predictions = rf.predict(X)

    df["Prediction"] = predictions
    df["Prediction"] = df["Prediction"].map(
        {0: "Legitimate", 1: "Fraudulent"}
    )

    st.markdown("### Prediction Results")
    st.dataframe(df.head(), use_container_width=True)

    fraud_count = (df["Prediction"] == "Fraudulent").sum()

    if fraud_count > 0:
        st.error(f"🚨 Fraud Transactions Detected: {fraud_count}")
    else:
        st.success("✅ No Fraud Transactions Detected")

else:
    st.info("🚀 Upload a CSV file to start prediction")

st.markdown('</div>', unsafe_allow_html=True)
