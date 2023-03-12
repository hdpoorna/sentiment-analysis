"""
py38
hdpoorna
check whether MASK_ZERO is supported by the models
"""

# import packages
import tensorflow as tf
from helpers import config
# from helpers.dataset import vectorize_layer, train_ds, val_ds
from helpers.dataset_dl import vectorize_layer, train_ds, val_ds
from helpers.models import AverageModel, RNNModel, TransformerModel


model = AverageModel(encoder=vectorize_layer, embedding_dim=config.EMBEDDING_DIM, dropouts=config.MODEL_DROPOUTS, deeper=True)
# model = RNNModel(encoder=vectorize_layer, embedding_dim=config.EMBEDDING_DIM, rnn_type="LSTM", rnn_stacked=False, dropouts=config.MODEL_DROPOUTS, deeper=True)
# model = TransformerModel(encoder=vectorize_layer, embedding_dim=config.EMBEDDING_DIM, dropouts=config.MODEL_DROPOUTS, deeper=True)

sample_text = "it was great."
prediction1 = model.predict([sample_text])
print(prediction1[0])

padding = "the " * 200
prediction2 = model.predict([sample_text, padding])
print(prediction2[0])

# should be the same
