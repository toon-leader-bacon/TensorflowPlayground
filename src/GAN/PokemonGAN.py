import tensorflow as tf


def tanh_bounded(input: float, min: float, max: float):
    # Effectively a linear interpolation between min and max
    # at the location given by the tanh(input)
    return (max - min) * tf.nn.tanh(input) + min
    # TODO: Consider forcing the value to conform to a near integer?
