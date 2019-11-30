from __future__ import absolute_import, division, print_function, unicode_literals
import functools

import numpy as np
import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split

heart_file = pd.read_csv("heart.csv")
heart_file.head()

train, test = train_test_split(heart_file, test_size=0.2)
train, val = train_test_split(train, test_size=0.2)

def df_to_dataset(dataframe, shuffle=True, batch_size=32):
    dataframe = dataframe.copy()
    labels = dataframe.pop('chd')
    ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(dataframe))
    ds = ds.batch(batch_size)
    return ds

batch_size = 5
train_ds = df_to_dataset(train, batch_size=batch_size)
val_ds = df_to_dataset(val, shuffle=False, batch_size=batch_size)
test_ds = df_to_dataset(test, shuffle=False, batch_size=batch_size)

feature_columns = []

# numeric cols
for header in ['sbp', 'tobacco', 'ldl', 'adiposity', 'typea','obesity','alcohol','age']:
    feature_columns.append(tf.feature_column.numeric_column(header))

# indicator cols
famhist = tf.feature_column.categorical_column_with_vocabulary_list(
    'famhist', ['Present', 'Absent'])
famhist_one = tf.feature_column.indicator_column(famhist)
feature_columns.append(famhist_one)

model = tf.keras.Sequential([
    tf.keras.layers.DenseFeatures(feature_columns),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(1, activation='sigmoid'),
    ])
    

model.compile(optimizer='adam',
                loss='binary_crossentropy',
                metrics=['accuracy'])

print("--Fit model--")
model.fit(train_ds, epochs=10)

print("--Evaluate model--")
model_loss, model_acc = model.evaluate(test_ds, verbose=2)
print(f"Model Loss:    {model_loss:.2f}")
print(f"Model Accuray: {model_acc*100:.1f}%")

# model.save('notMNIST.h5')