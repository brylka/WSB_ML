import tensorflow as tf
import numpy as np
from math import log, e

#Entropia rozkładu prawdopodobieństwa
np.set_printoptions(precision=6, suppress=True)
def entropy(labels, base=None):
    n_labels = len(labels)

    if n_labels <= 1:
        return 0

    value, counts = np.unique(labels, return_counts=True)
    probs = counts / n_labels
    n_classes = np.count_nonzero(probs)

    if n_classes <= 1:
        return 0

    ent = 0

    base = e if base is None else base
    for i in probs:
        ent -= i * log(i, base)
    return ent


labels = [1, 3, 5, 2, 3, 5, 3, 2, 1, 3, 4, 5]
print(entropy(labels))

labels = [3, 3, 3, 3, 3, 3, 3]
print(entropy(labels))



#Binarna entropia krzyżowa - Binary Crossentropy
from tensorflow.keras.losses import binary_crossentropy

y_true = np.array([1, 0, 1, 1, 0, 1], dtype='float')
y_pred = np.array([1, 0, 1, 1, 0, 1], dtype='float')

print(binary_crossentropy(y_true, y_pred))

# Kategoryczna entropia krzyżowa - Categorical Crossentropy
from tensorflow.keras.losses import categorical_crossentropy

y_true = np.array([1, 0, 1, 1, 2, 0, 1, 1, 2], dtype='float')
y_pred = np.array([1, 0, 1, 1, 2, 0, 1, 1, 2], dtype='float')

print(categorical_crossentropy(y_true, y_pred))

y_true = np.array([0, 1, 1, 1, 2, 1, 1, 1, 1], dtype='float')
y_pred = np.array([0, 1, 1, 1, 2, 1, 1, 1, 1], dtype='float')

print(categorical_crossentropy(y_true, y_pred))
