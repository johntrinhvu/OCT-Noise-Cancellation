# John Vu
# OCT Research, BLI
# 13 Jan 2025

from filtering import X_train, X_val, y_train, y_val, X_test, y_test
from model import unet_model
import tensorflow as tf

# assign higher weight to white pixels (corneal layer) in loss func
def weighted_binary_crossentropy(y_true, y_pred):
    weight_background = 0.2
    weight_foreground = 1.0
    weights = y_true * weight_foreground + (1 - y_true) * weight_background

    # binary crossentropy loss obj no reduction
    bce_fn = tf.keras.losses.BinaryCrossentropy(reduction=tf.keras.losses.Reduction.NONE)

    bce = bce_fn(y_true, y_pred)
    bce = tf.expand_dims(bce, axis=-1)
    weighted_bce = weights * bce
    return tf.reduce_mean(weighted_bce)

# init the UNet model
input_shape = (512, 1000, 1)
model = unet_model(input_shape)

# compile the model
model.compile(optimizer='adam', loss=weighted_binary_crossentropy, metrics=['accuracy'])

# train the model
# optimize # of epochs and batch size
# grid optimization
# Underfitting

# params
epochs = 20
batch_size = 1
accumulation_steps = 4

# data set
train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).batch(batch_size)
val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(batch_size)

# optimizer
optimizer = tf.keras.optimizers.Adam()

# loss obj
loss_object = tf.keras.losses.BinaryCrossentropy()

# training loop
for epoch in range(epochs):
    print(f"\nEpoch {epoch + 1}/{epochs}")

    # init accumulators for gradients and losses
    accumulated_gradients = [tf.zeros_like(var) for var in model.trainable_variables]
    epoch_loss = 0

    for step, (x_batch, y_batch) in enumerate(train_dataset):
        with tf.GradientTape() as tape:
            predictions = model(x_batch, training=True)
            loss = loss_object(y_batch, predictions) / accumulation_steps

        # compute gradients
        gradients = tape.gradient(loss, model.trainable_variables)

        # accum gradients
        for i, grad in enumerate(gradients):
            accumulated_gradients[i] += grad

        # apply gradients every 'accumulation_steps'
        if (step + 1) % accumulation_steps == 0 or (step + 1) == len(train_dataset):
            optimizer.apply_gradients(zip(accumulated_gradients, model.trainable_variables))

            # reset accumulators
            accumulated_gradients = [tf.zeros_like(var) for var in model.trainable_variables]

        epoch_loss += loss.numpy()
    print(f"Epoch {epoch + 1} Loss: {epoch_loss:.4f}")

    # validation
    val_loss = 0
    for x_batch, y_batch in val_dataset:
        predictions = model(x_batch, training=False)
        val_loss += loss_object(y_batch, predictions).numpy()

    print(f"Validation Loss: {val_loss / len(val_dataset):.4f}")

# history = model.fit(
#     X_train, y_train,
#     validation_data=(X_val, y_val),
#     epochs=20,
#     batch_size=4
# )

# save the trained model
model.save('oct_filter_model.h5')

# print training results
print("Training complete")

test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test Loss: {test_loss}")
print(f"Test Accuracy: {test_accuracy}")
