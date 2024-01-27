import numpy as np

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


img_train = np.load("train_imgs.npy")
y_train = np.load("train_labels.npy")

num_samples, height, width, channels = img_train.shape
img_flattened = img_train.reshape(num_samples, height * width * channels)
print(img_flattened.reshape((num_samples, height, width, channels)) == img_train)

print(img_flattened.shape, img_train.shape, y_train.shape)
print(img_flattened)

# for sets in img_train:
#     print(sets[0])
#     knn = KNeighborsClassifier(n_neighbors=4)
#     knn.fit(sets, y_train)

knn = KNeighborsClassifier(n_neighbors=4)
knn.fit(img_flattened, y_train)
