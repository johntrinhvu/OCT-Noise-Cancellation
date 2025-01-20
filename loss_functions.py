# John Vu
# OCT Research, BLI
# 13 Jan 2025


# loss function
def weighted_binary_crossentropy(y_true, y_pred):
    weight_background = 0.2
    weight_foreground = 1.0
    weights = y_true * weight_foreground + (1 - y_true) * weight_background
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    weighted_bce = weights * bce
    return tf.reduce_mean(weighted_bce)


# dice loss
def dice_loss(y_true, y_pred):
    numerator = 2 * tf.reduce_sum(y_true * y_pred)
    denominator = tf.reduce_sum(y_true + y_pred)
    return 1 - (numerator + 1e-7) / (denominator + 1e-7)


# combined loss
def combined_loss(y_true, y_pred):
    return 0.5 * weighted_binary_crossentropy(y_true, y_pred) + 0.5 * dice_loss(y_true, y_pred)
