# John Vu
# OCT Research, BLI
# 13 Jan 2025

import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model('oct_filter_model.h5')

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
noisy_image_path = 'path_to_noisy_image.bmp'
noisy_image = preprocess_image(noisy_image_path)
cleaned_image = model.predict(noisy_image)

# Save the cleaned image
cleaned_image = postprocess_image(cleaned_image)
cv2.imwrite('cleaned_image.bmp', cleaned_image)
print("Cleaned image saved!")
