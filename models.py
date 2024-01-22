import numpy as np
import tensorflow as tf

INPUT_SHAPE = (400, 275, 3)

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

    inputs = tf.keras.layers.Input(shape=INPUT_SHAPE)

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


def get_custom_conv(
    conv_activation: str = "relu",
    dense_activation: str = "relu"
):
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=INPUT_SHAPE),

        tf.keras.layers.Conv2D(64, kernel_size=(3, 3), padding="same", activation=conv_activation),
        tf.keras.layers.MaxPooling2D(pool_size=(3, 3)),

        tf.keras.layers.Conv2D(128, kernel_size=(3, 3), padding="same", activation=conv_activation),
        tf.keras.layers.MaxPooling2D(pool_size=(3, 3)),

        tf.keras.layers.Conv2D(256, kernel_size=(3, 3), padding="same", activation=conv_activation),
        tf.keras.layers.MaxPooling2D(pool_size=(3, 3)),

        tf.keras.layers.Flatten(),

        tf.keras.layers.Dense(100, activation=dense_activation),
        tf.keras.layers.Dense(400*275, activation="sigmoid"),
        tf.keras.layers.Reshape((400, 275, 1)),
        tf.keras.optimizers.RMSprop
    ])

    return model


def get_conv_only(conv_activation="relu"):
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=INPUT_SHAPE),

        tf.keras.layers.ZeroPadding2D(padding=(0, 288-275)),

        tf.keras.layers.Conv2D(64, kernel_size=(3, 3), padding="same", activation=conv_activation),
        tf.keras.layers.AveragePooling2D(pool_size=(2, 2)),

        tf.keras.layers.Conv2D(128, kernel_size=(3, 3), padding="same", activation=conv_activation),
        tf.keras.layers.AveragePooling2D(pool_size=(2, 2)),
        
        tf.keras.layers.Conv2D(256, kernel_size=(3, 3), padding="same", activation=conv_activation),
        tf.keras.layers.AveragePooling2D(pool_size=(2, 2)),
        
        tf.keras.layers.Conv2D(128, kernel_size=(3, 3), padding="same", activation=conv_activation),
        tf.keras.layers.UpSampling2D((2, 2)),
        
        tf.keras.layers.Conv2D(64, kernel_size=(3, 3), padding="same", activation=conv_activation),
        tf.keras.layers.UpSampling2D((2, 2)),
        
        tf.keras.layers.Conv2D(32, kernel_size=(3, 3), padding="same", activation=conv_activation),
        tf.keras.layers.UpSampling2D((2, 2)),

        tf.keras.layers.Conv2D(1, kernel_size=(3, 3), padding="same", activation="sigmoid"),
        tf.keras.layers.Cropping2D(((0, 0), (0, 21)))
    ])

    return model


def get_resnet_transfer() -> tf.keras.Model:
    def get_resnet_base(x):
        resnet = tf.keras.applications.resnet50.ResNet50(include_top=False, input_shape=INPUT_SHAPE)
        resnet.trainable = False
        return resnet(x)

    def get_upsample(x):
        x = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
        x = tf.keras.layers.UpSampling2D(size=(3, 3))(x)
        x = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='valid')(x)
        x = tf.keras.layers.UpSampling2D(size=(2, 2))(x)
        x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='valid')(x)
        x = tf.keras.layers.UpSampling2D(size=(2, 2))(x)
        x = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='valid')(x)
        x = tf.keras.layers.UpSampling2D(size=(3, 3))(x)
        x = tf.keras.layers.Cropping2D(((13, 13), (4, 3)))(x)
        return x

    inputs = tf.keras.layers.Input(shape=INPUT_SHAPE)
    preprocessed = tf.keras.applications.resnet50.preprocess_input(inputs)

    resnet_base = get_resnet_base(preprocessed)
    upsamples = get_upsample(resnet_base)
    outputs = tf.keras.layers.Conv2D(1, (3, 3), activation="sigmoid", padding="same")(upsamples)

    return tf.keras.Model(inputs=inputs, outputs=outputs)


def get_mobilenet_transfer() -> tf.keras.Model:
    def get_mobilenet_base(x):
        mobilenet = tf.keras.applications.MobileNetV3Large(include_top=False, input_shape=(224, 224, 3))
        mobilenet.trainable = False
        return mobilenet(x)

    def get_upsample(x):
        x = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='valid')(x)
        x = tf.keras.layers.UpSampling2D(size=(3, 3))(x)
        x = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='valid')(x)
        x = tf.keras.layers.UpSampling2D(size=(3, 3))(x)
        x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='valid')(x)
        x = tf.keras.layers.UpSampling2D(size=(4, 4))(x)
        x = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='valid')(x)
        x = tf.keras.layers.UpSampling2D(size=(3, 2))(x)
        
        x = tf.keras.layers.Cropping2D(((19, 19), (9, 8)))(x)
        return x

    inputs = tf.keras.layers.Input(shape=INPUT_SHAPE)
    resized = tf.keras.layers.Resizing(224, 224)(inputs)
    preprocessed = tf.keras.applications.mobilenet_v3.preprocess_input(resized)

    mobilenet_base = get_mobilenet_base(preprocessed)
    upsamples = get_upsample(mobilenet_base)
    outputs = tf.keras.layers.Conv2D(1, (3, 3), activation="sigmoid", padding="same")(upsamples)

    return tf.keras.Model(inputs=inputs, outputs=outputs)