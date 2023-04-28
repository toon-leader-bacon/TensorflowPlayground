import matplotlib.pyplot as plt
import os
import re
import shutil
import string
import tensorflow as tf
from keras import layers, utils, losses

print(tf.__version__)
# 3
# Get the data
data_src_url: str = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
dataset_path: str = utils.get_file("aclImdb_v1", data_src_url,
                                   untar=True, cache_dir='.',
                                   cache_subdir='')
dataset_dir: str = os.path.join(os.path.dirname(dataset_path), 'aclImdb')

# Take a peek in the downloaded data
# ['train', 'imdb.vocab', 'test', 'README', 'imdbEr.txt']
os.listdir(dataset_dir)

# ['neg',
#  'urls_neg.txt','urls_unsup.txt','unsupBow.feat','unsup','urls_pos.txt', 'pos', 'labeledBow.feat']
train_dir: str = os.path.join(dataset_dir, 'train')
os.listdir(train_dir)

# Take a look at a positive review
sample_file: str = os.path.join(train_dir, 'pos/1181_9.txt')
with open(sample_file) as f:
    print(f.read())

###################
# Prepare the data

# Use the Text Dataset form Directory tool
# https://www.tensorflow.org/api_docs/python/tf/keras/utils/text_dataset_from_directory
# Requires exactly two directories with data. One dir for each class

# Remove unsup directory
shutil.rmtree(os.path.join(train_dir, 'unsup'))

batch_size: int = 32
seed: int = 7654

# 25`000 files in 2 classes
# 20`000 files used for training
raw_train_ds: tf.data.Dataset = tf.keras.utils.text_dataset_from_directory(
    'aclImdb/train',
    batch_size=batch_size,  # default 32
    validation_split=0.2,  # % of data to reserve for validation
    subset='training',  # What subset of data to get 'training', 'validation', 'both
    seed=seed,  # seed for random shuffle etc.
    shuffle=True
)  # type: ignore
print(f"nocab type: {type(raw_train_ds)}")
# Let's take a look at these tf.data objects
# Each element in the data set contains a text and label batch
for text_batch, label_batch in raw_train_ds.take(1):
    for i in range(3):
        print("Review", text_batch.numpy()[i])  # text of the review
        print("Label", label_batch.numpy()[i])  # int of either 1 or 0

print("Label 0 corresponds to", raw_train_ds.class_names[0])  # 'neg'
print("Label 1 corresponds to", raw_train_ds.class_names[1])  # 'pos'

# 5`000 files used for validation
# Because we're using the seed there should be no overlap between the train and validation data
raw_val_ds: tf.data.Dataset = tf.keras.utils.text_dataset_from_directory(
    'aclImdb/train',
    batch_size=batch_size,
    validation_split=0.2,
    subset='validation',
    seed=seed,
    shuffle=True
)  # type: ignore

# All of the data
raw_test_ds: tf.data.Dataset = tf.keras.utils.text_dataset_from_directory(
    'aclImdb/test',
    batch_size=batch_size)  # type: ignore


def custom_standardization(input_data: str) -> str:
    # Clean the data (remove punctuation and/or html tags)
    # Convert data to all lowercase
    lowercase = tf.strings.lower(input_data)  # : tf.Tensor[str]
    # Remove html tags
    stripped_html: tf.Tensor = tf.strings.regex_replace(
        lowercase, '<br />', ' ')
    # Escape all punctuation
    return tf.strings.regex_replace(stripped_html,
                                    '[%s]' % re.escape(string.punctuation),
                                    '')


# Vectorize the data means converting each token into a number
max_features: int = 10000  # 10k
sequence_length: int = 250

# A layer to convert text data into  various representations that the
# model can more easily understand
vectorize_layer: layers.TextVectorization = layers.TextVectorization(
    standardize=custom_standardization,  # type: ignore
    max_tokens=max_features,
    output_mode='int',
    output_sequence_length=sequence_length)

# Make a text-only dataset (without labels), then call adapt
# NOTE It's important to only use the training data when using the adapt(...) func
# The identify function is simply being mapped across the raw training data
train_text = raw_train_ds.map(lambda x, y: x)
# Use the training_text and pass it through the vectorization layer
vectorize_layer.adapt(train_text)


