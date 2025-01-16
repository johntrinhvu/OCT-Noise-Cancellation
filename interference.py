# John Vu
# OCT Research, BLI
# 13 Jan 2025

import cv2
import numpy as np
from tensorflow.keras.models import load_model
import tensorflow as tf

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# loss function
def weighted_binary_crossentropy(y_true, y_pred):
    weight_background = 0.2
    weight_foreground = 1.0
    weights = y_true * weight_foreground + (1 - y_true) * weight_background
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    weighted_bce = weights * bce
    return tf.reduce_mean(weighted_bce)

# Load the trained model
model = load_model('oct_filter_model.h5', custom_objects={'weighted_binary_crossentropy': weighted_binary_crossentropy})

# Load and preprocess a new noisy image
def preprocess_image(image_path, img_size=(1000, 512)):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, img_size)
    img = img / 255.0  # Normalize
    img = img[np.newaxis, ..., np.newaxis]  # Add batch and channel dimensions
    return img

# Postprocess the output
def postprocess_image(pred):
    pred = (pred[0, ..., 0] * 255).astype(np.uint8)  # Remove batch, scale to [0, 255]
    return pred

# Predict on a new image
noisy_image_path = './dataset/input/015.bmp'
noisy_image = preprocess_image(noisy_image_path)
cleaned_image = model.predict(noisy_image)

# Save the cleaned image
cleaned_image = postprocess_image(cleaned_image)
cv2.imwrite('cleaned_image3.bmp', cleaned_image)
print("Cleaned image saved!")
