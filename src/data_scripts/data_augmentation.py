import tensorflow as tf
import numpy as np

from typing import List


class Augment:
    def __init__(self, func, prob=0.5):
        self.prob = prob
        self.func = func

    def __call__(self, *args):
        if np.random.random() < self.prob:
            return self.func(*args)
        if len(args) == 1:
            return args[0]
        return args
        

@tf.function
def horiz_flip_together(img_input: tf.Tensor, img_truth: tf.Tensor = None):
    img_input = tf.image.flip_left_right(img_input)
    img_truth = tf.image.flip_left_right(img_truth)
    return img_input, img_truth

@tf.function
def vert_flip_together(img_input: tf.Tensor, img_truth: tf.Tensor = None):
    img_input = tf.image.flip_up_down(img_input)
    img_truth = tf.image.flip_up_down(img_truth)
    return img_input, img_truth

@tf.function
def random_saturation(img_input: tf.Tensor):
    img_input = tf.image.random_saturation(img_input, 5, 10)
    return img_input

@tf.function
def random_brightness(img_input: tf.Tensor):
    img_input = tf.image.random_brightness(img_input, 0.2)
    return img_input


def augment_together_dataset(dataset: tf.data.Dataset, augments: List[Augment]) -> tf.data.Dataset:
    for augment in augments:
        dataset = dataset.map(augment, num_parallel_calls=tf.data.AUTOTUNE)
    return dataset