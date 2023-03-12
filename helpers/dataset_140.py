"""
py38
hdpoorna
for local Twitter dataset
"""

# import packages
import os
import re
import string
import numpy as np
import pandas as pd
import tensorflow as tf
from helpers import config


# paths
data_dir = "data"
dataset_name = "sentiment140"
train_csv_path = os.path.join(data_dir, dataset_name, "training.1600000.processed.noemoticon.csv")
test_csv_path = os.path.join(data_dir, dataset_name, "testdata.manual.2009.06.14.csv")

train_df = pd.read_csv(train_csv_path, header=None, low_memory=False, encoding='latin-1')
test_df = pd.read_csv(test_csv_path, header=None, low_memory=False, encoding='latin-1')

train_df.drop(columns=[1, 2, 3, 4], inplace=True)
test_df.drop(columns=[1, 2, 3, 4], inplace=True)
test_df = test_df[test_df[0] != 2]

train_df.rename(columns={0: "label", 5: "text"}, inplace=True)
test_df.rename(columns={0: "label", 5: "text"}, inplace=True)

train_df.at[(train_df["label"] == 4), "label"] = 1
test_df.at[(test_df["label"] == 4), "label"] = 1

full_train_ds = tf.data.Dataset.from_tensor_slices((train_df["text"], train_df["label"]))
raw_test_ds = tf.data.Dataset.from_tensor_slices((test_df["text"], test_df["label"]))

train_ds_len = full_train_ds.cardinality().numpy()

full_train_ds = full_train_ds.shuffle(train_ds_len, seed=config.SEED)

raw_train_ds = full_train_ds.take(int((1.0 - config.VAL_SPLIT)*train_ds_len))
raw_val_ds = full_train_ds.skip(int((1.0 - config.VAL_SPLIT)*train_ds_len))

raw_train_ds = raw_train_ds.batch(config.BATCH_SIZE)
raw_val_ds = raw_val_ds.batch(config.BATCH_SIZE)
raw_test_ds = raw_test_ds.batch(config.BATCH_SIZE)


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
