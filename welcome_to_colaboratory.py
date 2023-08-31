
Hand Written Digit Prediction- Classification analysis

the digit dataset consist of 8*8  pixel images of digits. the images attribute of the dataset stores 8x8 arrays of grayscale values for each image. we willuse these array to visualize the first 4 images. the target attribute of the dataset stores the digit each image represents

Import library
"""

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

"""import data"""

from sklearn.datasets import load_digits

df = load_digits()

_, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
for ax, image, label in zip(axes, df.images, df.target):
  ax.set_axis_off()
  ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
  ax.set_title("Training: %i" % label)

"""Data Processing"""

df.images.shape

df.images[0]

df.images[0].shape

len(df.images)

n_samples = len(df.images)
data  = df.images.reshape((n_samples, -1))

data[0]

data[0].shape

data.shape

"""Scaling image data"""

data.min()

data.max()

data = data/16

data.min()

data.max()

data[0]

"""Train Test split data"""

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(data,df.target, test_size = 0.3)

x_train.shape, x_test.shape, y_train.shape, y_test.shape

"""Random Forest Model"""

from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier()

rf.fit(x_train, y_train)

"""PredictTest Data"""

y_pred = rf.predict(x_test)

y_pred

"""model accuracy"""

from sklearn.metrics import confusion_matrix, classification_report

confusion_matrix(y_test, y_pred)

print(classification_report(y_test, y_pred))
