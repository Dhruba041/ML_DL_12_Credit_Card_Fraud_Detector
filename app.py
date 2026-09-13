import streamlit as st
import joblib
import numpy as np
import pickle
import pandas as pd
import keras


st.markdown(
    """
    <style>
    .stApp {
        background-color: #90EE90;
        color: brown;   /* default text color */
    }

    h1 {
        color: #1f77b4;  /* blue title */
    }

    label {
        color: #1f77b4 !important;
        font-weight: 600;
    }
    div[data-testid="stRadio"] div[role="radiogroup"] label span p {
        color: brown !important;
        font-weight: 600;
    }
    
    </style>
    """,
    unsafe_allow_html=True
)


pt = pickle.load(open("powertransformer_bs_cc_fraud.pkl", "rb"))
scaler = pickle.load(open("scaler_bs_cc_fraud.pkl", "rb"))

model = keras.models.load_model("dl_borderline_smote.keras") #for keras == 3.15.0
#DL Model
#model = pickle.load(open("ml_borderline_smote_xgb_model.pkl", "rb"))
#ML Model

st.markdown(
    "<h1 style='text-align: center;'>Credit Card Fraud Detection</h1>",
    unsafe_allow_html=True
)

#st.title("Credit Card Fraud Detection")
st.image("CC_img.png", use_container_width=True)

st.write("This app Classifies whether a transaction is fraudulent or not based on the input features provided by the user. Please fill in the PCA details below to get the prediction.")

Time = st.number_input("Time", min_value=0.0, value=0.0, step=1.0, format="%.1f")
V1 = st.number_input("V1", min_value=-100.0, value=0.0, step=0.1, format="%.15f")
V2 = st.number_input("V2", min_value=-100.0, value=0.0, step=0.1, format="%.15f")
V3 = st.number_input("V3", min_value=-100.0, value=0.0, step=0.1, format="%.15f")
V4 = st.number_input("V4", min_value=-100.0, value=0.0, step=0.1, format="%.15f")
V5 = st.number_input("V5", min_value=-100.0, value=0.0, step=0.1, format="%.15f")
V6 = st.number_input("V6", min_value=-100.0, value=0.0, step=0.1, format="%.15f")
V7 = st.number_input("V7", min_value=-100.0, value=0.0, step=0.1, format="%.15f")
V8 = st.number_input("V8", min_value=-100.0, value=0.0, step=0.1, format="%.15f")
V9 = st.number_input("V9", min_value=-100.0, value=0.0, step=0.1, format="%.15f")
V10 = st.number_input("V10", min_value=-100.0, value=0.0, step=0.1, format="%.15f")
V11 = st.number_input("V11", min_value=-100.0, value=0.0, step=0.1, format="%.15f")
V12 = st.number_input("V12", min_value=-100.0, value=0.0, step=0.1, format="%.15f")
V13 = st.number_input("V13", min_value=-100.0, value=0.0, step=0.1, format="%.15f")
V14 = st.number_input("V14", min_value=-100.0, value=0.0, step=0.1, format="%.15f")
V15 = st.number_input("V15", min_value=-100.0, value=0.0, step=0.1, format="%.15f")
V16 = st.number_input("V16", min_value=-100.0, value=0.0, step=0.1, format="%.15f")
V17 = st.number_input("V17", min_value=-100.0, value=0.0, step=0.1, format="%.15f")
V18 = st.number_input("V18", min_value=-100.0, value=0.0, step=0.1, format="%.15f")
V19 = st.number_input("V19", min_value=-100.0, value=0.0, step=0.1, format="%.15f")
V20 = st.number_input("V20", min_value=-100.0, value=0.0, step=0.1, format="%.15f")
V21 = st.number_input("V21", min_value=-100.0, value=0.0, step=0.1, format="%.15f")
V22 = st.number_input("V22", min_value=-100.0, value=0.0, step=0.1, format="%.15f")
V23 = st.number_input("V23", min_value=-100.0, value=0.0, step=0.1, format="%.15f")
V24 = st.number_input("V24", min_value=-100.0, value=0.0, step=0.1, format="%.15f")
V25 = st.number_input("V25", min_value=-100.0, value=0.0, step=0.1, format="%.15f")
V26 = st.number_input("V26", min_value=-100.0, value=0.0, step=0.1, format="%.15f")
V27 = st.number_input("V27", min_value=-100.0, value=0.0, step=0.1, format="%.15f")
V28 = st.number_input("V28", min_value=-100.0, value=0.0, step=0.1, format="%.15f")
Amount = st.number_input("Amount", min_value=0.0, value=0.0, step=1.0, format="%.2f")


if st.button("Classify Transaction"):

    feature_names = [
        'Time',
        'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9',
        'V10', 'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17',
        'V18', 'V19', 'V20', 'V21', 'V22', 'V23', 'V24', 'V25',
        'V26', 'V27', 'V28',
        'Amount'
    ]

    #input_data = np.array([[Time, V1, V2, V3, V4, V5, V6, V7, V8, V9, V10, V11, V12, V13, V14,
     #                       V15, V16, V17, V18, V19, V20, V21, V22, V23, V24, V25,
     #                       V26, V27, V28, Amount]])

    input_data = pd.DataFrame(
        [[
            Time,
            V1, V2, V3, V4, V5, V6, V7, V8, V9, V10,
            V11, V12, V13, V14, V15, V16, V17, V18,
            V19, V20, V21, V22, V23, V24, V25, V26,
            V27, V28,
            Amount
        ]],
        columns=feature_names
    )

    skewed_cols = ['V1',
                    'V2',
                    'V3',
                    'V5',
                    'V6',
                    'V7',
                    'V9',
                    'V10',
                    'V12',
                    'V14',
                    'V16',
                    'V17',
                    'V18',
                    'V20',
                    'V21',
                    'V22',
                    'V23',
                    'V24',
                    'V27',
                    'V28',
                    'Amount']

    input_data[skewed_cols] = pt.transform(input_data[skewed_cols])
    
    input_data_scaled = scaler.transform(input_data)    
    
    prediction = model.predict(input_data_scaled)
    predicted_class = (prediction > 0.3).astype(int)[0][0]
    #DL Model
    #predicted_class = int(prediction[0])
    #ML Model

    
    if predicted_class == 1:
        Prediction_text = "The transaction is classified as Fraudulent."
    else:
        Prediction_text = "The transaction is classified as Non-Fraudulent."

    if Prediction_text == "The transaction is classified as Fraudulent.":
        bg_color = "#ffebee"      # Light red
        border_color = "#d32f2f"  # Red
        text_color = "#b71c1c"    # Dark red
    else:
        bg_color = "#e8f5e9"      # Light green
        border_color = "#388e3c"  # Green
        text_color = "#1b5e20"    # Dark green

    st.markdown(
        f"""
        <div style="
            background-color:{bg_color};
            padding:15px;
            border-radius:10px;
            border:2px solid {border_color};
            color:{text_color};
            font-size:18px;
            font-weight:bold;
            text-align:center;
        ">

            Predicted Machine Failure Possibility: {Prediction_text}
        </div>
        """,
        unsafe_allow_html=True
    )

    #            Fraud Probability: {prediction}