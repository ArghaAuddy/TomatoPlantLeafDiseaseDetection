from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model('model/tomato_disease_detector.h5')

# Create test data generator
test_datagen = ImageDataGenerator(rescale=1./255)

# Path to validation folder (make sure it's correct)
test_dir = 'data/new_val'

# Create generator — use (256, 256) as model was trained on this
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(256, 256),  # ✅ Corrected
    batch_size=32,
    class_mode='categorical',
    shuffle=False  # For consistent predictions vs labels
)

# Evaluate the model
loss, accuracy = model.evaluate(test_generator)
print(f"✅ Accuracy on new_val: {accuracy:.4f}")
