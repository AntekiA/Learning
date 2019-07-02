import tensorflow as tf
import numpy as np


LABELS = [0, 1]
LABEL_COLUMN = 'is_anomaly'
CSV_COLUMNS = ['timestamp', 'value']
file_path = r"C:\Users\wh110\Desktop\research\AD\real_1.csv"

FEATURE_COLUMNS = [column for column in CSV_COLUMNS if column != LABEL_COLUMN]

def get_dataset(file_path):
  dataset = tf.data.experimental.make_csv_dataset(
      file_path,
      batch_size=12, # Artificially small to make examples easier to show.
      label_name=LABEL_COLUMN,
      na_value="?",
      num_epochs=1
	  )
  return dataset
  

raw_train_data = get_dataset(file_path)
print(raw_train_data)
#examples, labels = next(iter(raw_train_data)) # Just the first batch.
#print("EXAMPLES: \n", examples, "\n")
#print("LABELS: \n", labels)
