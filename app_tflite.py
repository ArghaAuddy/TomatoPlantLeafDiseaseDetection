import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import time
import platform
import psutil
import os

# --- Class Labels ---
class_names = [
    'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight',
    'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot',
    'Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
    'Tomato___Tomato_mosaic_virus', 'Tomato___healthy'
]

# --- Load TFLite Model ---
@st.cache_resource
def load_model():
    interpreter = tf.lite.Interpreter(model_path="model/mobilenet_quantized.tflite")
    interpreter.allocate_tensors()
    return interpreter

# --- Preprocess Image for TFLite Model ---
def preprocess_image(image, input_shape, input_dtype):
    image = image.resize((input_shape[1], input_shape[2]))
    image_np = np.array(image)

    if input_dtype == np.float32:
        image_np = image_np.astype(np.float32) / 255.0
    elif input_dtype == np.uint8:
        image_np = image_np.astype(np.uint8)

    return np.expand_dims(image_np, axis=0)

# --- Inference Function ---
def predict(image, interpreter):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    input_shape = input_details[0]['shape']
    input_dtype = input_details[0]['dtype']

    input_data = preprocess_image(image, input_shape, input_dtype)

    interpreter.set_tensor(input_details[0]['index'], input_data)

    start = time.time()
    interpreter.invoke()
    end = time.time()

    output_data = interpreter.get_tensor(output_details[0]['index'])[0]

    pred_idx = np.argmax(output_data)
    confidence = output_data[pred_idx]

    return pred_idx, confidence, (end - start) * 1000  # ms

# --- System Info ---
def get_device_info():
    return f"""
+-------------------------- SYSTEM INFO -----------------------------+
| CPU: Apple M2                                                     |
| RAM: {round(psutil.virtual_memory().total / (1024**3), 1)} GB                              |
| GPU: Apple M2                                                     |
| OS: {platform.system()} {platform.mac_ver()[0]}                                    |
+-------------------------------------------------------------------+
"""

# --- Streamlit UI ---
st.title("🍅 Tomato Leaf Disease Detection (TFLite)")
st.markdown("Upload an image of a tomato plant leaf to detect the disease using a quantized MobileNet model.")

# Upload section
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")

    # Resize for display (not for inference)
    image_disp = image.resize((300, 300))
    st.image(image_disp, caption="Uploaded Image", use_container_width=False)

    # Load model
    interpreter = load_model()

    # Predict
    pred_idx, confidence, inf_time = predict(image, interpreter)
    predicted_class = class_names[pred_idx]

    # Show ASCII output
    st.text(get_device_info())
    st.text(f"""
+-------------------------- MODEL INFERENCE ------------------------+
| Model: model/mobilenet_quantized.tflite                           |
| Predicted Class: {predicted_class.ljust(48)}|
| Confidence: {confidence:.2f}                                          |
| Inference Time (Single Image): {inf_time:.2f} ms                      |
+------------------------------------------------------------------+
""")
