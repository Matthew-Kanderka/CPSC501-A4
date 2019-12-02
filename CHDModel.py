from __future__ import absolute_import, division, print_function, unicode_literals
import functools

import numpy as np
import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow_core.python.keras import regularizers
from numpy.random import RandomState

LABEL_COLUMN = "chd"
TRAIN_DATA_PATH = "heart_train.csv"
TEST_DATA_PATH = "heart_test.csv"

df = pd.read_csv("heart.csv")
rng = RandomState()

train = df.sample(frac=0.80, random_state=rng)
test = df.loc[~df.index.isin(train.index)]

train.to_csv("heart_train.csv", index=False)
test.to_csv("heart_test.csv", index=False)

class PackNumericFeatures(object):
  def __init__(self, names):
    self.names = names

  def __call__(self, features, labels):
    numeric_features = [features.pop(name) for name in self.names]
    numeric_features = [tf.cast(feat, tf.float32) for feat in numeric_features]
    numeric_features = tf.stack(numeric_features, axis=-1)
    features['numeric'] = numeric_features

    return features, labels

def normalize_numeric_data(data, mean, std):
  # Center the data
  return (data-mean)/std

def get_dataset(file_path, **kwargs):
    dataset = tf.data.experimental.make_csv_dataset(file_path, batch_size=5, label_name=LABEL_COLUMN)
    return dataset

raw_train_data = get_dataset(TRAIN_DATA_PATH)
raw_test_data = get_dataset(TEST_DATA_PATH)

NUMERIC_FEATURES = ['sbp', 'tobacco', 'ldl', 'adiposity', 'typea','obesity','alcohol','age']

packed_train_data = raw_train_data.map(PackNumericFeatures(NUMERIC_FEATURES))
packed_test_data = raw_test_data.map(PackNumericFeatures(NUMERIC_FEATURES))

# normalize data
desc = pd.read_csv(TRAIN_DATA_PATH)[NUMERIC_FEATURES].describe()

MEAN = np.array(desc.T['mean'])
STD = np.array(desc.T['std'])

normalizer = functools.partial(normalize_numeric_data, mean=MEAN, std=STD)
numeric_column = tf.feature_column.numeric_column('numeric', normalizer_fn=normalizer, shape=[len(NUMERIC_FEATURES)])
numeric_columns = [numeric_column]

CATEGORIES = {
    'famhist': ['Present', 'Absent']
}

categorical_columns = []
for feature, vocab in CATEGORIES.items():
    cat_col = tf.feature_column.categorical_column_with_vocabulary_list(
        key=feature, vocabulary_list=vocab)
    categorical_columns.append(tf.feature_column.indicator_column(cat_col))

preprocessing_layer = tf.keras.layers.DenseFeatures(categorical_columns+numeric_columns)

def create_ds(dataframe, batch_size=1):
    dataframe = dataframe.copy()
    labels = dataframe.pop('chd')
    return tf.data.Dataset.from_tensor_slices((dict(dataframe), labels)).shuffle(buffer_size=len(dataframe)).batch(batch_size)

model = tf.keras.Sequential([
    preprocessing_layer,
    tf.keras.layers.Dense(16, kernel_regularizer=regularizers.l2(0.01), activation = 'elu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(16, kernel_regularizer=regularizers.l2(0.01), activation = 'elu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    

model.compile(optimizer='adamax',
                loss='binary_crossentropy',
                metrics=['accuracy'])

print("--Fit model--")
model.fit(packed_train_data, epochs=20, steps_per_epoch=128)

print("--Evaluate model--")
model_loss, model_acc = model.evaluate(packed_test_data, verbose=2, steps=128)
print(f"Model Loss:    {model_loss:.2f}")
print(f"Model Accuray: {model_acc*100:.1f}%")

# model.save('notMNIST.h5')