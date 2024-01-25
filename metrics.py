import tensorflow as tf


@tf.function
def SSIMLossF(y_true: tf.Tensor, y_pred: tf.Tensor):
    return 1 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, 1.0))


class SSIMLoss(tf.keras.losses.Loss):
    def __init__(self, name="SSIM", **kwargs):
        super().__init__(name=name, **kwargs)

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor):
        return SSIMLossF(y_true, y_pred)


@tf.function
def MSSSIM_LossF(y_true: tf.Tensor, y_pred: tf.Tensor):
    return (
        1 - tf.reduce_mean(tf.image.ssim_multiscale(y_true, y_pred, 1.0)) + 1e5
    )


class MSSSIM_Loss(tf.keras.losses.Loss):
    def __init__(self, name="MS-SSIM", **kwargs):
        super().__init__(name=name, **kwargs)

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor):
        return MSSSIM_LossF(y_true, y_pred)


@tf.function
def MSSSIM_L1_LossF(y_true: tf.Tensor, y_pred: tf.Tensor, alpha: float = 0.84):
    return alpha * MSSSIM_LossF(y_true, y_pred) + (
        1 - alpha
    ) * tf.keras.losses.mae(y_true, y_pred)


class MSSSIM_L1_Loss(tf.keras.losses.Loss):
    def __init__(self, alpha: float = 0.84, name="MS-SSIM_L1", **kwargs):
        super().__init__(name=name, **kwargs)
        self.alpha = alpha

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor):
        return MSSSIM_L1_LossF(y_true, y_pred)


@tf.function
def PSNRMetricF(y_true: tf.Tensor, y_pred: tf.Tensor):
    psnr_vals = tf.image.psnr(y_true, y_pred, 1.0)
    return tf.reduce_mean(psnr_vals)


class PSNRMetric(tf.keras.metrics.Metric):
    def __init__(self, name="psnr", **kwargs):
        super().__init__(name=name, **kwargs)
        self.psnr_sum = self.add_weight(name="psnr_sum", initializer="zeros")
        self.num_examples = self.add_weight(
            name="num_examples", initializer="zeros"
        )

    def update_state(
        self, y_true: tf.Tensor, y_pred: tf.Tensor, sample_weight=None
    ):
        psnr_vals = tf.image.psnr(y_true, y_pred, 1.0)
        self.psnr_sum.assign_add(tf.reduce_sum(psnr_vals))
        self.num_examples.assign_add(tf.cast(tf.shape(y_true)[0], tf.float32))

    def result(self):
        return self.psnr_sum / self.num_examples

    def reset_state(self):
        self.psnr_sum.assign(0.0)
        self.num_examples.assign(0.0)


class SSIMMetric(tf.keras.metrics.Metric):
    def __init__(self, name="ssim", **kwargs):
        super().__init__(name=name, **kwargs)
        self.ssim_sum = self.add_weight(name="ssim_sum", initializer="zeros")
        self.num_examples = self.add_weight(
            name="num_examples", initializer="zeros"
        )

    def update_state(
        self, y_true: tf.Tensor, y_pred: tf.Tensor, sample_weight=None
    ):
        psnr_vals = tf.image.ssim(y_true, y_pred, 1.0)
        self.ssim_sum.assign_add(tf.reduce_sum(psnr_vals))
        self.num_examples.assign_add(tf.cast(tf.shape(y_true)[0], tf.float32))

    def result(self):
        return self.ssim_sum / self.num_examples

    def reset_state(self):
        self.ssim_sum.assign(0.0)
        self.num_examples.assign(0.0)


# def PSNRMetric(y_true: tf.Tensor, y_pred: tf.Tensor):
#     return tf.reduce_mean(tf.image.psnr(y_true, y_pred, 1.0))
