import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model

# ------------------------------
# Load model
# ------------------------------
model = load_model('model/tomato_disease_detector_finetuned.h5')

# ------------------------------
# Load validation/test data
# ------------------------------
test_dir = 'data/test'  # keeping your path unchanged
img_size = (256, 256)

test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=img_size,
    batch_size=32,
    class_mode='categorical',
    shuffle=False  # Important for accurate label matching
)

# ------------------------------
# Predictions
# ------------------------------
preds = model.predict(test_generator)
y_pred = np.argmax(preds, axis=1)
y_true = test_generator.classes
class_labels = list(test_generator.class_indices.keys())

# ------------------------------
# Test Accuracy
# ------------------------------
test_accuracy = accuracy_score(y_true, y_pred)
print(f"\n✅ Test Accuracy: {test_accuracy * 100:.2f}%")

# Plot Test Accuracy
plt.figure(figsize=(4,5))
plt.bar(['Test Accuracy'], [test_accuracy * 100], color='green')
plt.ylim(0, 100)
plt.ylabel('Accuracy (%)')
plt.title('Test Accuracy')
plt.tight_layout()
plt.savefig("results/test_accuracy.png", dpi=300)
plt.show()

# ------------------------------
# Classification Report
# ------------------------------
print("\n🔹 Classification Report:")
print(classification_report(y_true, y_pred, target_names=class_labels))

# ------------------------------
# Confusion Matrix
# ------------------------------
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_labels, yticklabels=class_labels)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig("results/confusion_matrix_test.png", dpi=300) 
plt.show()
