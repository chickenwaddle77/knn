import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import seaborn as sns

# df = pd.read_csv('data.csv')
# X = df.drop('fraud', axis=1)
# y = df['fraud']

img_train = np.load("train_imgs.npy")
y_train = np.load("train_labels.npy")


# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Scale the features using StandardScaler so that both columns of the matrix are relative to each other
scaler = StandardScaler()

# Scale the training and test set so that the mean is 0? and S.D. is 1? (from i've read so far)
X_train = scaler.fit_transform(img_train)
# X_test = scaler.transform(X_test)

# Applying KNN model
knn = KNeighborsClassifier(n_neighbors=4)
knn.fit(img_train, y_train)

# Make prediction using the X set of data, the output is y_pred because we want to know the y set (0 or 1)
# y_pred = knn.predict(X_test)

# Compare what the actual y value is with the prediction for all values in the prediction set
# accuracy = accuracy_score(y_test, y_pred)
# print(y_test)
# print("Accuracy:", round(accuracy, 2))

sns.scatterplot(x=df['signature'],y=df['label'], hue=df['fraud'])
