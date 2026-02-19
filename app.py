# app.py

import streamlit as st
import numpy as np
import joblib

# ------------------------------
# Page Configuration
# ------------------------------
st.set_page_config(
    page_title="Order Status Predictor",
    page_icon="🚀",
    layout="centered",
)

# ------------------------------
# Load Model & Encoder
# ------------------------------
model = joblib.load("rf_model.pkl")
le = joblib.load("label_encoder.pkl")

# ------------------------------
# Sidebar Navigation
# ------------------------------
st.sidebar.title("📌 Menu")
page = st.sidebar.radio("Navigation", ["Home", "Prediction", "Dashboard", "Contact"])

# ------------------------------
# Home Section
# ------------------------------
if page == "Home":
    st.markdown(
        """
        <div style='text-align: center;'>
            <h1>Online Food Ordering Dashboard!</h1>
            <h3>Use this System to Predict The Status of orders based on delivery details.</h3>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Centered hero image using columns
    col1, col2, col3 = st.columns([0.1, 0.8, 0.1])
    with col2:
        st.image(
            r"C:\Users\user\Downloads\ibm-clg-project-new\ibm-clg-project\ibm\home.webp",
            caption="Online Food Ordering Home Page",
            width=800
        )


    # EDA Storyboard
    st.markdown("---")
    st.markdown("<h2>📌 Online Food Ordering EDA Storyboard</h2>", unsafe_allow_html=True)

    st.markdown(
        """
        **Problem Statement:**  
        Online food platforms struggle with cancellations, delays, peak hours, and customer behavior analysis.

        **Objective:**  
        To analyze order trends and prepare a storytelling-based EDA report.

        **Expected Output:**  
        - EDA graphs (peak times, cancellations, delays)  
        - Key insights explaining customer behavior  
        - Data storytelling presentation
        """
    )

# ------------------------------
# Prediction Section
# ------------------------------
elif page == "Prediction":
    st.markdown("<h1 style='text-align: center;'>🚀 Order Status Prediction</h1>", unsafe_allow_html=True)
    st.markdown("### Enter Order Details Below")

    with st.container():
        col1, col2 = st.columns(2)

        with col1:
            distance = st.slider("📍 Distance (km)", 0.0, 20.0, 3.0)
            bill_subtotal = st.number_input("💰 Bill Subtotal", min_value=0.0, value=300.0)
            packaging = st.number_input("📦 Packaging Charges", min_value=0.0, value=20.0)

        with col2:
            total = st.number_input("🧾 Total Amount", min_value=0.0, value=320.0)
            kpt = st.slider("⏱ KPT Duration (minutes)", 0, 120, 25)
            rider_wait = st.slider("🚴 Rider Wait Time (minutes)", 0, 60, 5)

    st.markdown("---")

    if st.button("🔍 Predict Order Status"):
        features = np.array([[distance, bill_subtotal, packaging, total, kpt, rider_wait]])
        prediction = model.predict(features)
        predicted_label = le.inverse_transform(prediction)[0]
        probabilities = model.predict_proba(features)[0]
        class_names = le.classes_
        confidence = np.max(probabilities) * 100

        # Display Results
        st.markdown("## 📊 Prediction Result")
        if predicted_label.lower() == "delivered":
            st.success(f"✅ Order Status: {predicted_label}")
        else:
            st.error(f"❌ Order Status: {predicted_label}")
            st.warning("⚠ This order is at risk (Not Delivered)")

        st.info(f"Model Confidence: {confidence:.2f}%")

        st.markdown("---")
        st.markdown("### 🔎 Model Confidence Distribution")
        for cls, prob in zip(class_names, probabilities):
            st.write(f"**{cls}** : {prob*100:.2f}%")
            st.progress(float(prob))

        if "Delivered" in class_names:
            delivered_index = list(class_names).index("Delivered")
            delivered_prob = probabilities[delivered_index] * 100
            risk = 100 - delivered_prob

            st.markdown("---")
            st.markdown("### 🚦 Delivery Risk Summary")
            st.write(f"Delivery Probability: {delivered_prob:.2f}%")
            st.write(f"Non-Delivery Risk: {risk:.2f}%")









                   # Centered hero image using columns
    col1, col2, col3 = st.columns([0.1, 0.8, 0.1])
    with col2:
        st.image(
            r"C:\Users\user\Downloads\ibm-clg-project-new\ibm-clg-project\ibm\delivery-boy.jpg",
            caption="delivery-Boy",
            width=800
        )











# ------------------------------
# Dashboard Section (Power BI Image)
# ------------------------------
elif page == "Dashboard":
    st.markdown("<h1 style='text-align: center;'>📈 Dashboard</h1>", unsafe_allow_html=True)
    st.markdown("### Power BI Dashboard Overview")

    st.image(
        r"C:\Users\user\Downloads\ibm-clg-project-new\ibm-clg-project\ibm\power bi_page-0001.jpg",
        caption="Power BI Dashboard",
        width=1000
    )

# ------------------------------
# Contact Section
# ------------------------------
elif page == "Contact":
    st.markdown("<h1 style='text-align: center;'>📞 Contact</h1>", unsafe_allow_html=True)
    st.markdown("For inquiries, please contact:")
    st.write("Email: team11@email.com")
    st.write("Phone: +91-1234567890")
