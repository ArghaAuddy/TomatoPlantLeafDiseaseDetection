import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import cv2
import os

# -----------------------------
# Load model and set image size
# -----------------------------
model = load_model('../model/mobilenet_tomato_model.h5')
img_size = (224, 224)

# -----------------------------
# Preprocess Image
# -----------------------------
def preprocess_image(img_path, size):
    img = image.load_img(img_path, target_size=size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    return img_array, np.array(img)

# -----------------------------
# Grad-CAM Heatmap
# -----------------------------
def get_gradcam_heatmap(model, img_array, last_conv_layer_name='Conv_1', pred_index=None):
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)

    return heatmap.numpy(), int(pred_index.numpy())

# -----------------------------
# Display & Save Grad-CAM
# -----------------------------
def display_gradcam(original_img, heatmap, predicted_class, img_name, alpha=0.4):
    img = cv2.resize(original_img, img_size)
    heatmap = cv2.resize(heatmap, img_size)

    heatmap = np.uint8(255 * heatmap)
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_img = cv2.addWeighted(heatmap_color, alpha, img, 1 - alpha, 0)

    # Show the image
    plt.imshow(cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB))
    plt.title(f"Grad-CAM: {predicted_class}")
    plt.axis('off')
    plt.show()

    # Save the output
    output_dir = "./gradcam_outputs"
    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(output_dir, f"gradcam_{img_name}.jpg")
    cv2.imwrite(output_path, superimposed_img)
    print(f"✅ Grad-CAM saved as: {output_path}")

# -----------------------------
# Dynamic Runner
# -----------------------------
def run_gradcam_on_image(img_path):
    # Load class names dynamically from validation folder
    val_dir = '../data/val'
    class_names = sorted(os.listdir(val_dir))

    # Preprocess image
    img_array, original_img = preprocess_image(img_path, img_size)

    # Generate heatmap and predict
    heatmap, pred_index = get_gradcam_heatmap(model, img_array)

    # Predict class
    predicted_class = class_names[pred_index] if pred_index < len(class_names) else f"Class {pred_index}"
    print("✅ Predicted Class:", predicted_class)

    # Extract image name
    img_name = os.path.splitext(os.path.basename(img_path))[0]

    # Display and Save
    display_gradcam(original_img.astype(np.uint8), heatmap, predicted_class, img_name)

# -----------------------------
# Example usage
# -----------------------------
# Replace this with any image you want to visualize Grad-CAM for
user_image_path = input("📸 Enter image path (e.g., ../data/val/ClassName/sample.jpg): ")
if os.path.exists(user_image_path):
    run_gradcam_on_image(user_image_path)
else:
    print("❌ Invalid image path. Please check the path and try again.")
