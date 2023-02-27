# https://towardsdatascience.com/train-your-own-chess-ai-66b9ca8d71e4
# 
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
    

# TODO: Get training data from Lichess
# TODO: Write code for training this NN
