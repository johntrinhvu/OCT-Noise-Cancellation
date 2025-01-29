# John Vu
# OCT Research, BLI
# 13 Jan 2025

from filtering import X_train, X_val, y_train, y_val, X_test, y_test
from model import unet_model
from loss_functions import weighted_binary_crossentropy, dice_loss, combined_loss
import tensorflow as tf


# init the UNet model
input_shape = (512, 1000, 1)
model = unet_model(input_shape)


# compile the model
model.compile(optimizer='adam', loss=combined_loss, metrics=['accuracy'])


# params
epochs = 20
batch_size = 1
accumulation_steps = 4


# data set
train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).batch(batch_size).prefetch(tf.data.AUTOTUNE)
val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(batch_size).prefetch(tf.data.AUTOTUNE)


# optimizer, loss object, and learning rate
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
loss_object = tf.keras.losses.BinaryCrossentropy()
current_lr = optimizer.learning_rate.numpy()

# training loop
for epoch in range(epochs):
    print(f"\nEpoch {epoch + 1}/{epochs}")

    # init accumulators for gradients and losses
    accumulated_gradients = [tf.zeros_like(var) for var in model.trainable_variables]
    epoch_loss = 0
    correct_predictions = 0
    total_predictions = 0

    for step, (x_batch, y_batch) in enumerate(train_dataset):
        # ensure y_batch == float32
        y_batch = tf.cast(y_batch, tf.float32)

        with tf.GradientTape() as tape:
            predictions = model(x_batch, training=True)
            loss = loss_object(y_batch, predictions) / accumulation_steps # normalize loss

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

        # update loss and accuracy
        epoch_loss += loss.numpy()
        correct_predictions += tf.reduce_sum(tf.cast(tf.equal(tf.round(predictions), y_batch), tf.float32)).numpy()
        total_predictions += tf.size(y_batch).numpy()
    
    epoch_accuracy = correct_predictions / total_predictions
    print(f"Epoch {epoch + 1} Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}")

    # validation
    val_loss = 0
    val_correct_predictions = 0
    val_total_predictions = 0

    for x_batch, y_batch in val_dataset:
        # ensure y_batch == float32
        y_batch = tf.cast(y_batch, tf.float32)

        predictions = model(x_batch, training=False)
        val_loss += loss_object(y_batch, predictions).numpy()
        val_correct_predictions += tf.reduce_sum(tf.cast(tf.equal(tf.round(predictions), y_batch), tf.float32)).numpy()
        val_total_predictions += tf.size(y_batch).numpy()

    val_accuracy = val_correct_predictions / val_total_predictions
    avg_val_loss = val_loss / len(val_dataset)
    print(f"Validation Loss: {val_loss / len(val_dataset):.4f}, Validation Accuracy: {val_accuracy:.4f}")

    # update the learning rate
    if epoch > 0 and avg_val_loss >= previous_val_loss:
        current_lr = max(current_lr * 0.5, 1e-6)
        optimizer.learning_rate.assign(current_lr)
        print(f"Reduced learning rate to current_lr:.6f")

    previous_val_loss = avg_val_loss


# save the trained model
model.save('oct_filter_model_1_27_25_with_dark_eyes.h5')

# print training results
print("Training complete")
