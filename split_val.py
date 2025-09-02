import os
import shutil
import random

original_val_path = 'data/val'
new_val_path = 'data/new_val'
test_path = 'data/test'

os.makedirs(new_val_path, exist_ok=True)
os.makedirs(test_path, exist_ok=True)

for cls in os.listdir(original_val_path):
    cls_path = os.path.join(original_val_path, cls)
    if not os.path.isdir(cls_path):
        continue

    images = os.listdir(cls_path)
    random.shuffle(images)

    split_index = len(images) // 2
    new_val_imgs = images[:split_index]
    test_imgs = images[split_index:]

    os.makedirs(os.path.join(new_val_path, cls), exist_ok=True)
    os.makedirs(os.path.join(test_path, cls), exist_ok=True)

    for img in new_val_imgs:
        src = os.path.join(cls_path, img)
        dest = os.path.join(new_val_path, cls, img)
        shutil.copy(src, dest)

    for img in test_imgs:
        src = os.path.join(cls_path, img)
        dest = os.path.join(test_path, cls, img)
        shutil.copy(src, dest)

print("✅ Split completed: new_val/ and test/ folders created.")
