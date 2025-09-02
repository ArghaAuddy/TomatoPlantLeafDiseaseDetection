import numpy as np
import tensorflow as tf
from PIL import Image

def load_tflite_model(path):
    interpreter = tf.lite.Interpreter(model_path=path)
    interpreter.allocate_tensors()
    return interpreter

def preprocess_image(image: Image.Image, target_size=(224, 224)):
    image = image.resize(target_size)
    img_array = np.array(image) / 255.0
    return np.expand_dims(img_array.astype(np.float32), axis=0)

def predict_tflite(interpreter, input_data):
    input_index = interpreter.get_input_details()[0]['index']
    output_index = interpreter.get_output_details()[0]['index']

    interpreter.set_tensor(input_index, input_data)
    interpreter.invoke()

    output_data = interpreter.get_tensor(output_index)
    return output_data[0]
