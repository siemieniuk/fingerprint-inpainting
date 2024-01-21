import tensorflow as tf


@tf.function
def invert(img: tf.Tensor):
    img = 255.0 - img
    return img

@tf.function
def to_grayscale(img: tf.Tensor):
    img = tf.image.rgb_to_grayscale(img)
    return img

@tf.function
def normalize_pixels(img: tf.Tensor):
    img = img / 255.0
    return img
