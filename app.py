import streamlit as st
import pickle
import numpy as np

# ---------- CSS: Background + Elevation ----------
st.markdown("""
<style>

/* Background image */
.stApp {
    background-image: url("https://images3.alphacoders.com/632/632929.jpg");
    background-size: cover;
    background-position: center;
    background-repeat: no-repeat;
}

/* Fade-in elevation animation */
@keyframes fadeIn {
    from {
        opacity: 0;
        transform: translateY(30px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

/* Main container (card) */
div.block-container {
    background-color: rgba(0, 0, 0, 0.40);
    color: white;
    border-radius: 18px;
    padding: 2.5rem;
    animation: fadeIn 1.5s ease-out;
    box-shadow: 0 20px 40px rgba(0, 0, 0, 0.6);
}

/* Title */
h1 {
    color: #FFD700;
    text-align: center;
}

/* Chemical labels */
label {
    color: #FFD700 !important;
    font-weight: 600;
}

/* Button elevation */
button[kind="primary"] {
    background-color: #FFD700;
    color: black;
    font-weight: bold;
    border-radius: 12px;
    padding: 0.6rem 1.2rem;
    box-shadow: 0 10px 25px rgba(255, 215, 0, 0.4);
}

</style>
""", unsafe_allow_html=True)

# ---------- LOAD MODEL ----------
with open("model_RF.pkl", "rb") as f:
    model = pickle.load(f)

try:
    with open("scalar.pkl", "rb") as f:
        scaler = pickle.load(f)
except:
    scaler = None

# ---------- UI ----------
st.title("🍷 Wine Quality Prediction")
st.write("Enter the chemical properties of the wine to predict its quality:")

fixed_acidity = st.number_input("Fixed Acidity")
volatile_acidity = st.number_input("Volatile Acidity")
citric_acid = st.number_input("Citric Acid")
residual_sugar = st.number_input("Residual Sugar")
chlorides = st.number_input("Chlorides")
free_sulfur = st.number_input("Free Sulfur Dioxide")
total_sulfur = st.number_input("Total Sulfur Dioxide")
density = st.number_input("Density")
ph = st.number_input("pH")
sulphates = st.number_input("Sulphates")
alcohol = st.number_input("Alcohol")

# ---------- PREDICTION WITH SPARKLES ----------
if st.button("Predict Quality"):
    with st.spinner("Predicting wine quality... 🍷"):
        data = np.array([[fixed_acidity, volatile_acidity, citric_acid,
                          residual_sugar, chlorides, free_sulfur,
                          total_sulfur, density, ph,
                          sulphates, alcohol]])

        if scaler:
            data = scaler.transform(data)

        prediction = model.predict(data)

    # ✨ Sparkles / Confetti Effect
    st.balloons()

    # Result
    st.success(f"🍷 Predicted Wine Quality: {prediction[0]}")





