import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import os

# Page settings
st.set_page_config(page_title="Tomato Leaf Disease Detector", layout="centered")

# Load model
MODEL_PATH = os.path.join("model", "tomato_disease_detector.h5")
model = load_model(MODEL_PATH)

# Class labels (modify if needed)
class_labels = ['Bacterial_spot', 'Early_blight', 'Late_blight', 'Leaf_Mold',
                'Septoria_leaf_spot', 'Spider_mites', 'Target_Spot',
                'Tomato_Yellow_Leaf_Curl_Virus', 'Tomato_mosaic_virus', 'Healthy']

# Title
st.title("🍅 Tomato Leaf Disease Detection")
st.markdown("Upload an image of a tomato leaf to predict the disease type.")

# File uploader
uploaded_file = st.file_uploader("Choose a tomato leaf image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Load and display the image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Preprocess the image
    img_size = (256, 256)
    image = image.resize(img_size)
    image_array = img_to_array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)

    # Predict
    prediction = model.predict(image_array)
    predicted_class = class_labels[np.argmax(prediction)]
    confidence = np.max(prediction) * 100

    # Display prediction
    st.markdown("### 🔍 Prediction:")
    st.success(f"**{predicted_class}** with {confidence:.2f}% confidence.")

    # Optional: Show full class probabilities
    st.markdown("### 📊 All Class Probabilities:")
    for i, label in enumerate(class_labels):
        st.write(f"{label}: {prediction[0][i]*100:.2f}%")

# Footer
st.markdown("---")
st.caption("Made with ❤️ using TensorFlow and Streamlit")
