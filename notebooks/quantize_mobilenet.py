import tensorflow as tf

# Load original Keras model
model = tf.keras.models.load_model("../model/mobilenet_tomato_model.h5")

# Create TFLite converter
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# Set optimization
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# Convert the model
tflite_model = converter.convert()

# Save the TFLite model
with open("../model/mobilenet_quantized.tflite", "wb") as f:
    f.write(tflite_model)

print("Quantized model saved as mobilenet_quantized.tflite")
