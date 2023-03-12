"""
py38
hdpoorna
"""

# import packages
import os
from datetime import datetime
import tensorflow as tf
from helpers import config
# from helpers.dataset import vectorize_layer, train_ds, val_ds, test_ds, dataset_name          # for local IMDb dataset
from helpers.dataset_dl import vectorize_layer, train_ds, val_ds, test_ds, dataset_name       # to download and use IMDb dataset
# from helpers.dataset_140 import vectorize_layer, train_ds, val_ds, test_ds, dataset_name        # for Twitter dataset
from helpers.models import AverageModel, RNNModel, TransformerModel
from helpers.write_results import save_graphs, model_summary_to_lines, write_to_txt, make_dir


model = AverageModel(encoder=vectorize_layer, embedding_dim=config.EMBEDDING_DIM, dropouts=config.MODEL_DROPOUTS, deeper=True)
# model = RNNModel(encoder=vectorize_layer, embedding_dim=config.EMBEDDING_DIM, rnn_type="LSTM", rnn_stacked=False, dropouts=config.MODEL_DROPOUTS, deeper=True)
# model = TransformerModel(encoder=vectorize_layer, embedding_dim=config.EMBEDDING_DIM, dropouts=config.MODEL_DROPOUTS, deeper=True)

model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
              optimizer=tf.keras.optimizers.Adam(config.LEARNING_RATE),
              metrics=tf.metrics.BinaryAccuracy(threshold=0.5))

# train
history = model.fit(train_ds,
                    epochs=config.EPOCHS,
                    validation_data=val_ds)

# predict
sample = ["it was great.", "it was ok.", "it was terrible."]
prediction = model.predict(sample)
print(prediction)

# test
test_loss, test_acc = model.evaluate(test_ds)
print('Test Loss:', test_loss)
print('Test Accuracy:', test_acc)

# make model id
now_utc = datetime.utcnow()
now_str = now_utc.strftime("%Y-%m-%d-%H-%M-%S")
model_id = "{}-{}-{}".format(model.name, dataset_name, now_str)

# save graphs
save_graphs(history_dict=history.history, model_id=model_id)

# write to text file
model_summary = model_summary_to_lines(model)
write_to_txt(model_id=model_id,
             model_summary=model_summary,
             history_dict=history.history,
             test_acc=test_acc,
             test_loss=test_loss)

# save model
make_dir("saved_models")
model_dir_path = os.path.join("saved_models", "{}-model".format(model_id))
model.compile(optimizer="adam")
tf.saved_model.save(model, model_dir_path)
print("model saved to {}".format(model_dir_path))


