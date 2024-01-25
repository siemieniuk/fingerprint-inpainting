import tensorflow as tf
from typing import List, Callable


class ImageDatasetFactory:
    def __init__(self, batch_size: int = 4, seed: int = 42):
        self.image_size = (400, 275)
        self.batch_size = batch_size
        self.seed = seed

    def __call__(self, images_path: str):
        return self.create_dataset(images_path)

    def create_dataset(self, images_path: str) -> tf.data.Dataset:
        ds = tf.keras.preprocessing.image_dataset_from_directory(
            images_path,
            seed=self.seed,
            batch_size=self.batch_size,
            labels=None,
            image_size=self.image_size,
            shuffle=False
        )
        return ds
    

class FingerprintDatasetFactory:
    def __init__(self, input_func_list: List[Callable], truth_func_list: List[Callable], batch_size: int = 4, seed: int = 42):
        @tf.function
        def input_transform(img):
            for input_func in input_func_list:
                img = input_func(img)
            return img

        @tf.function        
        def truth_transform(img):
            for truth_func in truth_func_list:
                img = truth_func(img)
            return img

        self.input_transform = input_transform
        self.truth_transform = truth_transform

        # self.input_func_list = input_func_list
        # self.truth_func_list = truth_func_list

        self.img_ds_factory = ImageDatasetFactory(batch_size=batch_size, seed=seed)

    def __call__(self, input_imgs_path: str, truth_imgs_path: str):
        return self.create_dataset(input_imgs_path, truth_imgs_path)

    def create_dataset(self, input_imgs_path: str, truth_imgs_path: str):
        input_ds = self.img_ds_factory(input_imgs_path)
        truth_ds = self.img_ds_factory(truth_imgs_path)

        input_ds = input_ds.map(self.input_transform, num_parallel_calls=tf.data.AUTOTUNE)
        truth_ds = truth_ds.map(self.truth_transform, num_parallel_calls=tf.data.AUTOTUNE)

        # for func in self.input_func_list:
        #     input_ds = input_ds.map(func, num_parallel_calls=tf.data.AUTOTUNE)
        # for func in self.truth_func_list:
        #     truth_ds = truth_ds.map(func, num_parallel_calls=tf.data.AUTOTUNE)

        all_ds = tf.data.Dataset.zip((input_ds, truth_ds))
        return all_ds