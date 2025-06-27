import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from PIL import Image

# Load your trained model
model = load_model(r"C:\Users\Muskan Singh\Desktop\MACHINE LEARNING\bestmodel.h5")

# App title
st.title("🧠 Brain Tumor Detector")
st.markdown("Upload a brain MRI scan to check for tumor presence.")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# Predict button
if uploaded_file is not None:
    # Display the uploaded image
    st.image(uploaded_file, caption="Uploaded MRI Scan", use_column_width=True)

    # Preprocess the image
    img = Image.open(uploaded_file).resize((224, 224))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Make prediction
    prediction = model.predict(img_array)[0][0]

    # Show result
    if prediction > 0.5:
        st.success("✅ No Brain Tumor Detected.")
    else:
        st.error("🧠 Brain Tumor Detected.")