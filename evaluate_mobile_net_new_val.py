import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
import os

# ------------------------------
# Load MobileNet model
# ------------------------------
model = load_model('model/mobilenet_tomato_model.h5')

# ------------------------------
# Test data generator
# ------------------------------
test_dir = 'data/new_val'
img_size = (224, 224)

test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=img_size,
    batch_size=32,
    class_mode='categorical',
    shuffle=False
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
test_acc = np.mean(y_pred == y_true)
print(f"✅ Test Accuracy: {test_acc*100:.2f}%")

# ------------------------------
# Classification Report
# ------------------------------
print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=class_labels))

# ------------------------------
# Confusion Matrix
# ------------------------------
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=class_labels, yticklabels=class_labels)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("MobileNet - Confusion Matrix on Test Data")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
os.makedirs("results", exist_ok=True)
plt.savefig("results/mobilenet_confusion_matrix_test.png", dpi=300)
plt.show()

# ------------------------------
# Test Accuracy and Loss Graph
# ------------------------------
# Compute test loss using evaluate
test_loss, test_accuracy = model.evaluate(test_generator)
print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")

# Plot test accuracy and loss
plt.figure(figsize=(6,5))
plt.bar(['Test Accuracy', 'Test Loss'], [test_accuracy, test_loss], color=['green','red'])
plt.ylim(0, 1)
plt.title('MobileNet Test Metrics')
for i, v in enumerate([test_accuracy, test_loss]):
    plt.text(i, v + 0.02, f"{v:.2f}", ha='center', fontweight='bold')
plt.savefig("results/mobilenet_test_metrics.png", dpi=300)
plt.show()
