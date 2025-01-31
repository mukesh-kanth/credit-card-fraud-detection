import streamlit as st
import numpy as np
import pandas as pd
import pickle

# Load the trained model
@st.cache_data
def load_model():
    with open("random.pkl", "rb") as file:
        random = pickle.load(file)
    return random

# Load the model
random = load_model()

# Streamlit UI
st.title("Credit Card Fraud Detection")
st.write("Enter transaction details to detect fraud.")

# Input fields for transaction data
amount = st.number_input("Transaction Amount", min_value=0.01, step=0.01)

# Assuming the model was trained on PCA-transformed data, user needs to input additional Vs
V1 = st.number_input("V1")
V2 = st.number_input("V2")
V3 = st.number_input("V3")
V4 = st.number_input("V4")
V5 = st.number_input("V5")
V6 = st.number_input("V6")
V7 = st.number_input("V7")
V8 = st.number_input("V8")
V9 = st.number_input("V9")
V10= st.number_input("V10")
V11= st.number_input("V11")
V12= st.number_input("V12")
V13= st.number_input("V13")
V14= st.number_input("V14")
V15= st.number_input("V15")
V16= st.number_input("V16")
V17= st.number_input("V17")
V18= st.number_input("V18")
V19= st.number_input("V19")
V20= st.number_input("V20")
V21= st.number_input("V21")
V22= st.number_input("V22")
V23= st.number_input("V23")
V24= st.number_input("V24")
V25= st.number_input("V25")
V26= st.number_input("V26")
V27= st.number_input("V27")
V28= st.number_input("V28")



# Create DataFrame from input
input_data = pd.DataFrame({
    "Time": [time],
    "Amount": [amount],
    "V1": [V1],
    "V2": [V2],
    "V3": [V3],
    "V4": [V4],
    "V5": [V5],
    "V6": [V6],
    "V7": [V7],
    "V8": [V8],
    "V9": [V9],
    "V10": [V10],
    "V11": [V11],
    "V12": [V12],
    "V12": [V12],
    "V13": [V13],
    "V14": [V14],
    "V15": [V15],
    "V16": [V16],
    "V17": [V17],
    "V18": [V18],
    "V19": [V19],
    "V20": [V20],
    "V21": [V21],
    "V22": [V22],
    "V23": [V23],
    "V24": [V24],
    "V25": [V25],
    "V26": [V26],
    "V27": [V27],
    "V28": [V28]
   
})

# Predict fraud
if st.button("DETECT"): 
    pred = random.predict(np.array([[float(V) for V in [V1,V2,V3,V4,V5,V6,V7,V8,V9,V10,V11,V12,V13,V14,V15,V16,V17,V18,V19,V20,V21,V22,V23,V24,V25,V26,V27,V28]]]))
    if pred[0] == 1:
        st.error("🚨 Fraudulent Transaction Detected!")
    else:
        st.success("✅ Transaction is Legitimate.")
