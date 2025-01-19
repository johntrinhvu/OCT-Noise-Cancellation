# John Vu
# OCT Research, BLI
# 13 Jan 2025

"""
CHANGE THE FILE DIRECTORY FOR NOISY PATH IMAGE TO WHICHEVER DIRECTORY 
THE IMAGE YOU WANT TO CLEAN IS IN ON LINE 47,
AND CHANGE THE FILE NAME OF THE IMAGE THAT WILL BE CREATED ON LINE 54.
"""

import cv2
import numpy as np
from tensorflow.keras.models import load_model
from loss_functions import weighted_binary_crossentropy, dice_loss, combined_loss
import tensorflow as tf

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# Load the trained model
model = load_model(
    'oct_filter_model4.h5', 
    custom_objects={
        'weighted_binary_crossentropy': weighted_binary_crossentropy,
        'dice_loss': dice_loss,
        'combined_loss': combined_loss
    }
)

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
    kernel = np.ones((3, 3), np.uint8)
    pred = cv2.morphologyEx(pred, cv2.MORPH_CLOSE, kernel)
    return pred

# Predict on a new image
noisy_image_path = './dataset/input/165.bmp'
noisy_image = preprocess_image(noisy_image_path)
cleaned_image = model.predict(noisy_image)

# Save the cleaned image
cleaned_image = postprocess_image(cleaned_image)
cv2.imwrite('./cleaned_images/fourth_model_attempts/cleaned_image14.bmp', cleaned_image)
print("Cleaned image saved!")
