import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import mean_squared_error
from math import sqrt

X = np.load("train_imgs.npy")
y = np.load("train_labels.npy")

img_train, img_test, y_train, y_test = train_test_split(X, y, test_size=0.22, random_state=42, stratify=y)

num_samples, height, width, channels = img_train.shape
# print(img_train.shape, y_train.shape)
img_train = img_train.reshape(num_samples, height * width * channels)
# print(img_train.shape, y_train.shape, img_train[:, 0].shape, img_train[:, 1].shape)
train_score = {}
test_score = {}

n_neighbors = 3

knn = KNeighborsClassifier(n_neighbors)
knn.fit(img_train, y_train)
acc = knn.score(img_train, y_train)
print(acc * 100)


# weights???? try that
# neighbors vs accuracy


# train_preds = knn.predict(img_train)

# mse = mean_squared_error(y_train, train_preds)
# rmse = sqrt(mse)
#
# print(rmse)

# cmap = sns.cubehelix_palette(as_cmap=True)  # some colour stuff???
# f, ax = plt.subplots()  # what is f and ax
# points = ax.scatter(img_train[:, 0], img_train[:, 1], c=train_preds, s=50, cmap=cmap)
# # x, y, c = gradient colourbar, cmap = size of dots
#
# f.colorbar(points)
# plt.show()








# train_score.update({neighbor: knn.score(img_train, y_train)})
# test_score.update({neighbor: knn.score(img_test, y_test)})

# print(train_score)
# print(test_score)
# plt.plot(n_neighbors, train_score, label="Train Accuracy")
# plt.plot(n_neighbors, test_score, label="Test Accuracy")
# plt.xlabel("Number Of Neighbors")
# plt.ylabel("Accuracy")
# plt.title("KNN: Varying number of Neighbors")
# plt.legend()
# plt.xlim(0, 33)
# plt.ylim(0.60, 0.90)
# plt.grid()
# plt.show()
#
# print(img_flattened.reshape((num_samples, height, width, channels)) == img_train)
#
# print(img_flattened.shape, img_train.shape, y_train.shape)
# print(img_flattened)

# for sets in img_train:
#     print(sets[0])
#     knn = KNeighborsClassifier(n_neighbors=4)
#     knn.fit(sets, y_train)
#
# knn = KNeighborsClassifier(n_neighbors=4)
# knn.fit(img_flattened, y_train)
