# John Vu
# OCT Research, BLI
# 13 Jan 2025

import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from loss_functions import weighted_binary_crossentropy, dice_loss, combined_loss
import tensorflow as tf

"""
Change the folder inputs on line 78 and 79, and then run the program
"""

# Configure the GPU memory growth
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


# Process all images in a folder:
def process_images(input_folder, output_folder, img_size=(1000, 512)):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        input_path = os.path.join(input_folder, filename)

        # check valid image file
        if os.path.isfile(input_path) and filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            print(f"Processing: {filename}")

            # preprocess the image, predict, and postprocess
            noisy_image = preprocess_image(input_path)
            cleaned_image = model.predict(noisy_image)
            cleaned_image = postprocess_image(cleaned_image)

            # save cleaned image to output folder
            output_path = os.path.join(output_folder, filename)
            cv2.imwrite(output_path, cleaned_image)
            print(f"Saved cleaned image to: {output_path}")


# Specify input and output folders
input_folder = './cleaned_images/test_interference'
output_folder = './cleaned_images/test_interference_output'

# Process images
process_images(input_folder, output_folder)
print("All images have been processed and saved.")
