# John Vu
# OCT Research, BLI
# 13 Jan 2025

import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

# paths:
input_dir = "dataset/input"
target_dir = "dataset/target"


# params:
img_size = (1000, 512) # resizing all images to this size


# loading of the images
def load_images(img_dir, img_size):
    images = []

    for img_name in sorted(os.listdir(img_dir)):
        img_path = os.path.join(img_dir, img_name)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE) # load as a grayscale

        if img is None:
            print(f"W: Failed to load {img_path}")
            continue

        print(f"Loaded {img_name}")
        img = cv2.resize(img, img_size)
        images.append(img)

    return np.array(images)


# load both the inp and clean images
try:
    input_images = load_images(input_dir, img_size)
    target_images = load_images(target_dir, img_size)

except Exception as e:
    print(f"Error during image loading: {e}")
    raise


# normal the pixel vals to range [0, 1]
input_images = input_images / 255.0
target_images = target_images / 255.0


# extr dimension, tensorflow/pytorch
input_images = input_images[..., np.newaxis]
target_images = target_images[..., np.newaxis]


# Split the data set:
X_train, X_temp, y_train, y_temp = train_test_split(
    input_images, target_images, test_size=0.3, random_state=42
)


X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42
)


# 70% training, 15% validation, 15% testing
print(f"Training set: {X_train.shape}, {y_train.shape}")
print(f"Validation set: {X_val.shape}, {y_val.shape}")
print(f"Test set: {X_test.shape}, {y_test.shape}")
