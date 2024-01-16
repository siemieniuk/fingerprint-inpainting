import tensorflow as tf

class FingerprintDatasetFactory:
    def __init__(self):
        self.image_size = (400, 275)

    def create_dataset(self, input_path: str, true_output_path: str, batch_size: int = 4, seed=42):
        input_ds = tf.keras.preprocessing.image_dataset_from_directory(
            input_path,
            seed=seed,
            batch_size=batch_size,
            labels=None,
            image_size=self.image_size,
            shuffle=False
        )

        truth_ds = tf.keras.preprocessing.image_dataset_from_directory(
            true_output_path,
            seed=seed,
            batch_size=batch_size,
            labels=None,
            image_size=self.image_size,
            shuffle=False   
        )

        all_ds = tf.data.Dataset.zip((input_ds, truth_ds))
        return all_ds