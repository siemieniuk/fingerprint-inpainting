import numpy as np
import tensorflow as tf


def postprocess_image(result) -> np.ndarray:
    res = tf.keras.utils.array_to_img(result)
    res = res.copy()
    res = np.repeat(res, repeats=3, axis=-1)
    res = ((1 - res) * 255.0).astype(np.uint8)
    return res
