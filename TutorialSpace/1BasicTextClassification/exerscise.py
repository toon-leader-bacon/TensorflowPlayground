import tensorflow as tf
from keras import layers
from typing import List

# https://www.tensorflow.org/tutorials/keras/text_classification#exercise_multi-class_classification_on_stack_overflow_questions

# Data found here: https://storage.googleapis.com/download.tensorflow.org/data/stack_overflow_16k.tar.gz

# Train a text classifier with 4 classes
# Stack overflow questions with language tags
# Tags are 'csharp', 'java', 'javascript', python'

class_names: list[str] = [
    'csharp',      # 0
    'java',        # 1
    'javascript',  # 2
    'python'       # 3
]

# Number of words in vocabulary
max_features: int = 10000  # 10k


def build_model() -> tf.keras.Sequential:
    embedding_dim: int = 16
    return tf.keras.Sequential([
        # Embedding layer for working with text
        # REMEMBER to vectorize the data before feeding into training
        # Embedding values for words are learned during training
        # input_dim = size of vocabulary, max integer index + 1
        # output_dim = size of dense embedding
        # Resultant dimension = (batch, sequence, embedding)
        layers.Embedding(input_dim=max_features + 1, output_dim=embedding_dim),
        layers.Dropout(0.2),

        # Average over the sequence dimension. Allows handling variable length input
        layers.GlobalAveragePooling1D(),
        layers.Dropout(0.2),

        # Dense layer for neural network magic
        tf.keras.layers.Dense(128, activation='relu'),
        layers.Dropout(0.2),

        # Output layer. Each node represents the % confidence of label
        tf.keras.layers.Dense(len(class_names))  # 4 classes
    ])


def build_vectorization_layer() -> layers.TextVectorization:
    sequence_length: int = 250
    result: layers.TextVectorization = layers.TextVectorization(
        standardize="lower_and_strip_punctuation",
        split="whitespace",  # Optional
        max_tokens=max_features,
        output_mode='int',
        # Pad output to be of dim: (batch_size, output_sequence_length)
        # Only valid with output_mode='int'
        output_sequence_length=sequence_length
    )
    return result


def vectorize_text_(text: str, label: str, vectorizer: layers.TextVectorization):
    # Each token will be converted to a number, and put in a tensor array
    text = tf.expand_dims(text, -1)
    return vectorizer(text), label


def vectorize_text(text: List[str], label: str) -> list[tuple]:
    vectorize_layer: layers.TextVectorization = build_vectorization_layer()
    result: list[tuple] = []  # list of tuple of (vectorizedText, label)
    for txt in text:
        result.append(vectorize_text_(txt, label, vectorize_layer))
    return result
