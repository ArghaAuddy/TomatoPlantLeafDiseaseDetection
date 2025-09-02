import streamlit as st
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import tensorflow as tf
import cv2

# Page configuration
st.set_page_config(page_title="Tomato Leaf Disease Detector", layout="centered")

# Global style and color scheme
st.markdown("""
    <style>
    .stApp {
        background-color: #000000;
        font-family: 'Segoe UI', sans-serif;
        color: #F2F0EF;
    }
    h1, h2, h3, h4, h5, h6, p, div {
        color: #F2F0EF !important;
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown(
    """
    <h1 style='text-align: center; color: tomato; font-size: 60px;'>
        🍅 Tomato Leaf Disease Detection
    </h1>
    """,
    unsafe_allow_html=True
)

st.markdown("<p style='text-align: center;'>Upload a tomato plant leaf image. The model will predict the disease and highlight the affected regions using Grad-CAM.</p>", unsafe_allow_html=True)
st.markdown("---")

# Load trained model
model = load_model("model/mobilenet_tomato_model.h5")

# Class labels
class_names = [
    'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight',
    'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
    'Tomato___Tomato_mosaic_virus', 'Tomato___healthy'
]

# Grad-CAM generation
def get_gradcam(image_array, model, last_conv_layer_name='Conv_1'):
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(image_array)
        predicted_class = tf.argmax(predictions[0])
        loss = predictions[:, predicted_class]
    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs), axis=-1)
    heatmap = np.maximum(heatmap, 0)
    heatmap /= tf.math.reduce_max(heatmap)
    heatmap = heatmap.numpy()
    heatmap = cv2.resize(heatmap, (224, 224))
    return heatmap

def overlay_heatmap(heatmap, image):
    heatmap = np.uint8(255 * heatmap)
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    image_np = np.array(image.resize((224, 224)))
    superimposed_img = cv2.addWeighted(image_np, 0.6, heatmap_color, 0.4, 0)
    return Image.fromarray(superimposed_img)

# File uploader
uploaded_file = st.file_uploader("Choose a leaf image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load and preprocess image
    image = Image.open(uploaded_file).convert('RGB')
    img_size = (224, 224)
    image_resized = image.resize(img_size)
    image_array = img_to_array(image_resized) / 255.0
    image_array = np.expand_dims(image_array, axis=0)

    # Prediction
    prediction = model.predict(image_array)
    predicted_class = class_names[np.argmax(prediction)]
    confidence = round(100 * np.max(prediction), 2)

    # Grad-CAM
    heatmap = get_gradcam(image_array, model)
    gradcam_image = overlay_heatmap(heatmap, image)

    # Side-by-side display
    st.markdown("### 🔍 Uploaded Image vs Grad-CAM")
    col1, col2 = st.columns(2)
    with col1:
        st.image(image_resized, caption="Original (Resized)", use_container_width=True)
    with col2:
        st.image(gradcam_image, caption="Grad-CAM", use_container_width=True)

    # Prediction results
    st.markdown("### 🧠 Prediction Details")
    st.success(f"**Disease:** {predicted_class}")
    st.info(f"**Confidence:** {confidence}%")
    st.progress(int(confidence))

    st.markdown("---")
    st.caption("Model: MobileNetV2 | Image Size: 224x224 | Powered by TensorFlow & Streamlit")
