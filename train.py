# John Vu
# OCT Research, BLI
# 13 Jan 2025

from filtering import X_train, X_val, y_train, y_val, X_test, y_test
from model import unet_model
import tensorflow as tf

# init the UNet model
input_shape = (512, 1000, 1)
model = unet_model(input_shape)

# compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# train the model
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