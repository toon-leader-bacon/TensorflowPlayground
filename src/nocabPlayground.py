import keras


def blab():
    inputs: keras.Input = keras.Input(shape=(3,))  # 3 dimensional input
    hidden1: keras.layers.Dense = keras.layers.Dense(
        units=32,  # dimensionality of the output space,
        activation='relu',  # activation function
        use_bias=True
    )(inputs)
    hidden2: keras.layers.Dense = keras.layers.Dense(
        units=16,
        activation='tanh',
        use_bias=False
    )(hidden1)
    output: keras.layers.Dense = keras.layers.Dense(
        units=5,
        activation='sigmoid'
    )(hidden2)

    model: keras.Model = keras.Model(inputs=inputs, outputs=output)
    model.summary()


blab()
