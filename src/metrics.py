import tensorflow as tf


def iou_metric(y_true, y_pred):
    y_pred = tf.cast(y_pred > 0.5, tf.float32)
    intersection = tf.reduce_sum(y_true * y_pred)
    union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) - intersection
    return intersection / (union + 1e-6)


def focal_loss(alpha=0.25, gamma=2.0):

    def loss(y_true, y_pred):

        y_true = tf.cast(y_true, tf.float32)

        # Compute BCE manually without reducing channel
        bce = tf.keras.backend.binary_crossentropy(y_true, y_pred)

        p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)

        focal = alpha * tf.pow((1 - p_t), gamma) * bce

        return tf.reduce_mean(focal)

    return loss