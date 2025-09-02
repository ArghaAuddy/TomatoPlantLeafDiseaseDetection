import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
import os

# Load MobileNet model
model = load_model('model/mobilenet_tomato_model.h5')

# Define test data directory
test_dir = 'data/test'
img_size = (224, 224)  # MobileNet expects 224x224 input

# Create ImageDataGenerator
test_datagen = ImageDataGenerator(rescale=1./255)

# Create test generator
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=img_size,
    batch_size=32,
    class_mode='categorical',
    shuffle=False  # Keep same order for ground truth
)

# Predict using model
preds = model.predict(test_generator)
y_pred = np.argmax(preds, axis=1)
y_true = test_generator.classes
class_labels = list(test_generator.class_indices.keys())

# Classification report
print("Classification Report:")
print(classification_report(y_true, y_pred, target_names=class_labels))

# Confusion matrix
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=class_labels, yticklabels=class_labels)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("MobileNet - Confusion Matrix on new_val")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()

# Save confusion matrix plot
os.makedirs("results", exist_ok=True)
plt.savefig("results/mobilenet_confusion_matrix_test.png", dpi=300)
plt.show()
