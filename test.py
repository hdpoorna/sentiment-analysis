"""
py38
hdpoorna
"""

# import packages
import os
import tensorflow as tf

model_id = "Average-aclImdb-2023-02-24-04-10-16"
model_dir_path = os.path.join("saved_models", "{}-model".format(model_id))

model = tf.saved_model.load(model_dir_path)

sample = ["it was great.", "it was ok.", "it was terrible."]
prediction = model(sample)
print(model_id)
print(prediction)


