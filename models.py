import numpy as np
import tensorflow as tf


def get_unet(
    depth: int = 3, dropout: float = 0.2, activation: str = "relu"
) -> tf.keras.Model:
    def get_downsample(x, n_filters: int, dropout: float = 0.2):
        conv_layer = tf.keras.layers.Conv2D(
            filters=n_filters,
            kernel_size=(3, 3),
            padding="same",
            activation=activation,
            kernel_initializer="he_normal",
        )(x)
        conv_layer = tf.keras.layers.Dropout(dropout)(conv_layer)
        conv_layer = tf.keras.layers.Conv2D(
            filters=n_filters,
            kernel_size=(3, 3),
            padding="same",
            activation=activation,
            kernel_initializer="he_normal",
        )(conv_layer)
        pool_layer = tf.keras.layers.MaxPooling2D((2, 2))(conv_layer)
        return conv_layer, pool_layer

    def get_upsample(x, conv_features, n_filters: int, dropout: float = 0.2):
        x = tf.keras.layers.Conv2DTranspose(
            filters=n_filters,
            kernel_size=(3, 3),
            strides=(2, 2),
            padding="same",
        )(x)

        # fix dimensions
        out_pad = [[0, 0], [0, 0]]
        cf_shape_comp = np.array(conv_features.shape[1:])
        x_shape = np.array(x.shape[1:])
        if cf_shape_comp[1] != x_shape[1]:
            out_pad[1][0] = 1
        if cf_shape_comp[0] != x_shape[0]:
            out_pad[0][0] = 1
        x = tf.keras.layers.ZeroPadding2D(padding=out_pad)(x)

        # concatenation and other operations
        x = tf.keras.layers.Concatenate()([x, conv_features])
        x = tf.keras.layers.Conv2D(
            filters=n_filters,
            kernel_size=(3, 3),
            padding="same",
            activation=activation,
            kernel_initializer="he_normal",
        )(x)
        x = tf.keras.layers.Dropout(dropout)(x)
        x = tf.keras.layers.Conv2D(
            filters=n_filters,
            kernel_size=(3, 3),
            padding="same",
            activation=activation,
            kernel_initializer="he_normal",
        )(x)

        return x

    def get_bottleneck(x, n_filters: int, dropout: float = 0.2):
        bottleneck = tf.keras.layers.Conv2D(
            filters=current_filters,
            kernel_size=(3, 3),
            activation=activation,
            padding="same",
        )(x)
        bottleneck = tf.keras.layers.Dropout(0.1)(bottleneck)
        bottleneck = tf.keras.layers.Conv2D(
            filters=current_filters,
            kernel_size=(3, 3),
            activation=activation,
            padding="same",
        )(bottleneck)

        return bottleneck

    assert depth >= 1

    inputs = tf.keras.layers.Input(shape=(400, 275, 3))

    current_filters = 16
    c, p = get_downsample(inputs, current_filters)
    current_filters *= 2

    contractions = [c]
    pools = [p]

    # downsampling
    for i in range(depth - 1):
        c, p = get_downsample(pools[i], current_filters)
        contractions.append(c)
        pools.append(p)
        current_filters *= 2

    # bottleneck
    bottleneck = get_bottleneck(pools[-1], current_filters)

    # upsampling
    u = get_upsample(bottleneck, contractions[-1], current_filters)
    current_filters //= 2
    upsamples = [u]

    for i in range(depth - 1):
        u = get_upsample(
            upsamples[i],
            contractions[depth - i - 2],
            current_filters,
        )
        upsamples.append(u)
        current_filters //= 2

    # output
    outputs = tf.keras.layers.Conv2D(
        filters=1,  # monochromatic image
        kernel_size=(1, 1),
        padding="same",
        activation="sigmoid",
    )(upsamples[-1])

    return tf.keras.Model(inputs=inputs, outputs=outputs, name="unet")
