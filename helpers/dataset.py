"""
py38
hdpoorna
for local IMDb dataset
"""

# import packages
import os
import re
import string
import numpy as np
import tensorflow as tf
from helpers import config


# paths
data_dir = "data"
dataset_name = "aclImdb"
train_dir = os.path.join(data_dir, dataset_name, "train")
test_dir = os.path.join(data_dir, dataset_name, "test")

# load ds
raw_train_ds = tf.keras.utils.text_dataset_from_directory(train_dir,
                                                          batch_size=config.BATCH_SIZE,
                                                          validation_split=config.VAL_SPLIT,
                                                          subset="training",
                                                          seed=config.SEED)

raw_val_ds = tf.keras.utils.text_dataset_from_directory(train_dir,
                                                        batch_size=config.BATCH_SIZE,
                                                        validation_split=config.VAL_SPLIT,
                                                        subset="validation",
                                                        seed=config.SEED)

raw_test_ds = tf.keras.utils.text_dataset_from_directory(test_dir,
                                                         batch_size=config.BATCH_SIZE)

# class names
print("Label 0 corresponds to", raw_train_ds.class_names[0])
print("Label 1 corresponds to", raw_train_ds.class_names[1])


def custom_standardization(input_data):
    lowercase = tf.strings.lower(input_data)
    stripped_html = tf.strings.regex_replace(lowercase, '<br />', ' ')
    return tf.strings.regex_replace(stripped_html,
                                    '[%s]' % re.escape(string.punctuation),
                                    '')


vectorize_layer = tf.keras.layers.TextVectorization(
    standardize=custom_standardization,
    max_tokens=config.MAX_FEATURES,
    output_mode='int',
    output_sequence_length=config.SEQUENCE_LENGTH)

# Make a text-only dataset (without labels), then call adapt
vectorize_layer.adapt(raw_train_ds.map(lambda text, label: text))

vocab = np.array(vectorize_layer.get_vocabulary())
print('Vocabulary size: {}'.format(len(vocab)))


# print(vocab[:20])


def vectorize_text(text, label):
    text = tf.expand_dims(text, -1)
    return vectorize_layer(text), label


# train_ds = raw_train_ds.map(vectorize_text)

AUTOTUNE = tf.data.AUTOTUNE

# train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
train_ds = raw_train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = raw_val_ds.cache().prefetch(buffer_size=AUTOTUNE)
test_ds = raw_test_ds.cache().prefetch(buffer_size=AUTOTUNE)

if __name__ == "__main__":
    # print some
    print("train\n")
    for text_batch, label_batch in raw_train_ds.take(1):
        for i in range(int(np.min([8, config.BATCH_SIZE]))):
            print("Original: ", text_batch[i].numpy())
            print()
            print("Round-trip: ", " ".join(np.squeeze(vocab[vectorize_text(text_batch[i], None)[0].numpy()])))
            print("Label", label_batch.numpy()[i])
            print()

    # print(vectorize_text(text_batch[0], label_batch[0]))

    print("val\n")
    for text_batch, label_batch in raw_val_ds.take(1):
        for i in range(int(np.min([8, config.BATCH_SIZE]))):
            print("Original: ", text_batch[i].numpy())
            print("Round-trip: ", " ".join(np.squeeze(vocab[vectorize_text(text_batch[i], None)[0].numpy()])))
            print("Label", label_batch.numpy()[i])
            print()
