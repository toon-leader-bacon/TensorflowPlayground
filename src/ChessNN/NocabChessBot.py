# https://towardsdatascience.com/train-your-own-chess-ai-66b9ca8d71e4
from tensorflow.keras import layers
import tensorflow as tf


def build_evaluation_model():
    model = tf.keras.Sequential()
    for layerN in range(4):  # TODO: Variable number of layers
        model.add(layers.Dense(
            808,  # dimensionality of the output space
            use_bias=False,
            input_shape=(808,)))
        model.add(layers.LeakyReLU())
        # model.add(layers.ReLU());

    # output layer:
    model.add(layers.Dense(
        1,  # dimensionality of the output space
        use_bias=False,
        input_shape=(808,)))


def build_loss_function():
    # TODO: Read about huber loss, and l1 loss below
    return tf.keras.losses.huber_loss  # Stackoverflow says this is essentially the same as L1


HUBER_DELTA = 0.5
def l1_loss_function(y_true, y_pred):
    x = tf.keras.backend.abs(y_true - y_pred)
    x = tf.keras.backend.switch(
        x < HUBER_DELTA,                       # if true
        0.5 * x ** 2,                          # Then x = this expression
        HUBER_DELTA * (x - 0.5 * HUBER_DELTA)  # else this expression
    )


def blab():
    model = build_evaluation_model()
    model.compile(optimizer='adam',
                  loss=l1_loss_function,
                  metrics=['accuracy'])


# TODO: Get training data from Lichess
# TODO: Write code for training this NN
