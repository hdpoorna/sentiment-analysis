"""
py38
hdpoorna
"""

# import packages
import numpy as np
import tensorflow as tf
from helpers import config


def AverageModel(encoder, embedding_dim=config.EMBEDDING_DIM, dropouts=config.AVG_MODEL_DROPOUTS, deeper=False):
    model = tf.keras.Sequential([encoder,
                                 tf.keras.layers.Embedding(len(encoder.get_vocabulary()) + 1,
                                                           embedding_dim,
                                                           # Use masking to handle the variable sequence lengths
                                                           mask_zero=config.MASK_ZERO)],
                                name="Average")
    if dropouts[0]["apply"]:
        model.add(tf.keras.layers.Dropout(dropouts[0]["rate"]))

    model.add(tf.keras.layers.GlobalAveragePooling1D())

    if dropouts[1]["apply"]:
        model.add(tf.keras.layers.Dropout(dropouts[1]["rate"]))

    if deeper:
        model.add(tf.keras.layers.Dense(embedding_dim, activation='relu'))
        if dropouts[2]["apply"]:
            model.add(tf.keras.layers.Dropout(dropouts[2]["rate"]))

    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

    print(model.summary())
    print("layers that support masking: ", [layer.supports_masking for layer in model.layers])

    return model


def RNNModel(encoder, embedding_dim=config.EMBEDDING_DIM, rnn_type="LSTM", rnn_stacked=False,
             dropouts=config.RNN_MODEL_DROPOUTS,
             deeper=False):
    model = tf.keras.Sequential([encoder,
                                 tf.keras.layers.Embedding(len(encoder.get_vocabulary()) + 1,
                                                           embedding_dim,
                                                           # Use masking to handle the variable sequence lengths
                                                           mask_zero=config.MASK_ZERO)],
                                name=rnn_type)
    if dropouts[0]["apply"]:
        model.add(tf.keras.layers.Dropout(dropouts[0]["rate"]))

    if rnn_type == "RNN":
        model.add(tf.keras.layers.Bidirectional(tf.keras.layers.SimpleRNN(embedding_dim, return_sequences=rnn_stacked)))
        if rnn_stacked:
            model.add(tf.keras.layers.Bidirectional(tf.keras.layers.SimpleRNN(
                np.power(2, np.max([0, np.floor(np.log2(embedding_dim) - 1)])).astype(int)
            )))
    elif rnn_type == "GRU":
        model.add(tf.keras.layers.Bidirectional(tf.keras.layers.GRU(embedding_dim, return_sequences=rnn_stacked)))
        if rnn_stacked:
            model.add(tf.keras.layers.Bidirectional(tf.keras.layers.GRU(
                np.power(2, np.max([0, np.floor(np.log2(embedding_dim) - 1)])).astype(int)
            )))
    elif rnn_type == "LSTM":
        model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(embedding_dim, return_sequences=rnn_stacked)))
        if rnn_stacked:
            model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(
                np.power(2, np.max([0, np.floor(np.log2(embedding_dim) - 1)])).astype(int)
            )))
    else:
        exit("choose RNN, GRU or LSTM as rnn_type of RNNModel")

    if dropouts[1]["apply"]:
        model.add(tf.keras.layers.Dropout(dropouts[1]["rate"]))

    if deeper:
        model.add(tf.keras.layers.Dense(embedding_dim, activation='relu'))
        if dropouts[2]["apply"]:
            model.add(tf.keras.layers.Dropout(dropouts[2]["rate"]))

    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

    print(model.summary())
    print("layers that support masking: ", [layer.supports_masking for layer in model.layers])

    return model


def positional_encoding(length=config.MAX_SEQUENCE_LENGTH, depth=config.EMBEDDING_DIM):
    depth = depth / 2

    positions = np.arange(length)[:, np.newaxis]  # (seq, 1)
    depths = np.arange(depth)[np.newaxis, :] / depth  # (1, depth)

    angle_rates = 1 / (10000 ** depths)  # (1, depth)
    angle_rads = positions * angle_rates  # (pos, depth)

    pos_encoding = np.concatenate(
        [np.sin(angle_rads), np.cos(angle_rads)],
        axis=-1)

    return tf.cast(pos_encoding, dtype=tf.float32)


class PositionalEmbedding(tf.keras.layers.Layer):
    def __init__(self, vocab_size, max_seq_len=config.MAX_SEQUENCE_LENGTH, embedding_dim=config.EMBEDDING_DIM):
        super().__init__()
        self.supports_masking = config.MASK_ZERO
        self.embedding_dim = embedding_dim
        self.embedding = tf.keras.layers.Embedding(input_dim=vocab_size + 1,
                                                   output_dim=embedding_dim,
                                                   # Use masking to handle the variable sequence lengths
                                                   mask_zero=config.MASK_ZERO
                                                   )
        self.pos_encoding = positional_encoding(length=max_seq_len, depth=embedding_dim)

    def compute_mask(self, *args, **kwargs):
        return self.embedding.compute_mask(*args, **kwargs)

    def call(self, x):
        seq_len = tf.shape(x)[-1]
        x = self.embedding(x)
        # This factor sets the relative scale of the embedding and positonal_encoding.
        x *= tf.math.sqrt(tf.cast(self.embedding_dim, tf.float32))
        x = x + self.pos_encoding[tf.newaxis, :seq_len, :]
        return x


class TokenAndPositionEmbedding(tf.keras.layers.Layer):
    def __init__(self, vocab_size, max_seq_len=config.MAX_SEQUENCE_LENGTH, embedding_dim=config.EMBEDDING_DIM):
        super().__init__()
        self.supports_masking = config.MASK_ZERO
        self.token_emb = tf.keras.layers.Embedding(input_dim=vocab_size + 1,
                                                   output_dim=embedding_dim,
                                                   # Use masking to handle the variable sequence lengths
                                                   mask_zero=config.MASK_ZERO
                                                   )
        self.pos_emb = tf.keras.layers.Embedding(input_dim=max_seq_len + 1,
                                                 output_dim=embedding_dim,
                                                 # Use masking to handle the variable sequence lengths
                                                 # mask_zero=config.MASK_ZERO
                                                 )

    def call(self, x):
        seq_len = tf.shape(x)[-1]
        positions = tf.range(start=0, limit=seq_len, delta=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions

    def compute_mask(self, *args, **kwargs):
        return self.token_emb.compute_mask(*args, **kwargs)


class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, embedding_dim=config.EMBEDDING_DIM, num_attn_heads=config.NUM_ATTN_HEADS,
                 feed_fwd_dim=config.FEED_FWD_DIM, dropout_rate=0.1):
        super().__init__()
        self.supports_masking = config.MASK_ZERO
        self.att = tf.keras.layers.MultiHeadAttention(num_heads=num_attn_heads, key_dim=embedding_dim)
        self.ffn = tf.keras.Sequential(
            [tf.keras.layers.Dense(feed_fwd_dim, activation="relu"), tf.keras.layers.Dense(embedding_dim)]
        )
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(dropout_rate)
        self.dropout2 = tf.keras.layers.Dropout(dropout_rate)

    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)


def TransformerModel(encoder, embedding_dim=config.EMBEDDING_DIM, max_seq_len=config.MAX_SEQUENCE_LENGTH,
                     num_attn_heads=config.NUM_ATTN_HEADS, feed_fwd_dim=config.FEED_FWD_DIM,
                     dropouts=config.TRANSFORMER_MODEL_DROPOUTS,
                     tf_block_dropout_rate=0.1, deeper=False):
    model = tf.keras.Sequential([encoder,
                                 TokenAndPositionEmbedding(       # TokenAndPositionEmbedding or PositionalEmbedding
                                     vocab_size=len(encoder.get_vocabulary()),
                                     max_seq_len=max_seq_len,
                                     embedding_dim=embedding_dim
                                 )],
                                name="Transformer")

    if dropouts[0]["apply"]:
        model.add(tf.keras.layers.Dropout(dropouts[0]["rate"]))

    model.add(TransformerBlock(embedding_dim=embedding_dim,
                               num_attn_heads=num_attn_heads,
                               feed_fwd_dim=feed_fwd_dim, dropout_rate=tf_block_dropout_rate))
    model.add(tf.keras.layers.GlobalAveragePooling1D())

    if dropouts[1]["apply"]:
        model.add(tf.keras.layers.Dropout(dropouts[1]["rate"]))

    if deeper:
        model.add(tf.keras.layers.Dense(embedding_dim, activation='relu'))
        if dropouts[2]["apply"]:
            model.add(tf.keras.layers.Dropout(dropouts[2]["rate"]))

    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

    print(model.summary())
    print("layers that support masking: ", [layer.supports_masking for layer in model.layers])

    return model
