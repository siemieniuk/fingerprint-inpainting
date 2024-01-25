import tensorflow as tf

from models import get_unet


def tune_model_mse(hp):
    hp_act = hp.Choice("activation", values=["relu", "elu", "tanh"])
    hp_depth = hp.Choice("depth", values=[2, 3, 4, 5])
    hp_dropout = hp.Float("dropout", min_value=0.05, max_value=0.2)
    hp_optimizer = hp.Choice("optimizer", values=["adam", "SGD", "rmsprop"])

    model = get_unet(depth=hp_depth, activation=hp_act, dropout=hp_dropout)
    match hp_optimizer:
        case "adam":
            optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        case "SGD":
            optimizer = optimizer = tf.keras.optimizers.SGD(
                learning_rate=0.001
            )
        case "rmsprop":
            optimizer = optimizer = tf.keras.optimizers.RMSprop(
                learning_rate=0.001
            )

    model.compile(optimizer=optimizer, loss="mse")
    return model
