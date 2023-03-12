"""
py38
hdpoorna
Analyze IMDb dataset and find a rough MAX_SEQUENCE_LENGTH
Need to set BATCH_SIZE = 1 in helpers/config.py
"""

## NEED TO SET BATCH_SIZE = 1

# import packages
import numpy as np
import tensorflow as tf
# from helpers.dataset import vectorize_layer, raw_train_ds
from helpers.dataset_dl import vectorize_layer, raw_train_ds

train_ds = raw_train_ds.map(lambda text, label: vectorize_layer(text))

train_ds_len = train_ds.cardinality().numpy()
print(train_ds_len)
arr = np.zeros(train_ds_len)

i = 0
for vector in train_ds.as_numpy_iterator():
    arr[i] = vector.shape[1]
    i += 1
    print(i, vector.shape[1])

maximum = np.max(arr)
print(maximum)
