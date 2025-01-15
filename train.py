# John Vu
# OCT Research, BLI
# 13 Jan 2025

from filtering import X_train, X_val, y_train, y_val, X_test, y_test
from model import unet_model
import tensorflow as tf

# init the UNet model
input_shape = (512, 1000, 1)
model = unet_model(input_shape)

# assign higher weight to white pixels (corneal layer) in loss func
def weighted_binary_crossentropy(y_true, y_pred):
    weight_background = 0.2
    weight_foreground = 1.0
    weights = y_true * weight_foreground + (1 - y_true) * weight_background
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    weighted_bce = weights * bce
    return tf.reduce_mean(weighted_bce)

# compile the model
model.compile(optimizer='adam', loss='weighted_binary_crossentropy', metrics=['accuracy'])

# train the model
# optimize # of epochs and batch size
# grid optimization
# Underfitting
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=20,
    batch_size=16
)

# save the trained model
model.save('oct_filter_model.h5')

# print training results
print("Training complete")

test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test Loss: {test_loss}")
print(f"Test Accuracy: {test_accuracy}")
