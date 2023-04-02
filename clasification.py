import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tensorflow.keras.datasets.mnist import load_data
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout

np.set_printoptions(precision=12, suppress=True, linewidth=120)

(X_train, y_train), (X_test, y_test) = load_data()

print(f'X_train shape: {X_train.shape}')
print(f'y_train shape: {y_train.shape}')
print(f'X_test shape: {X_test.shape}')
print(f'y_test shape: {y_test.shape}')

#print(y_train[0])

# Tworzenie modelu
model = Sequential()
model.add(Flatten(input_shape=(28, 28)))
model.add(Dense(units=128, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(units=10, activation='softmax'))

# Kompilacja modelu
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

print(model.summary())

# Trenowanie modelu
history = model.fit(X_train, y_train, epochs=15)

model.evaluate(X_test, y_test, verbose=2)

y_pred = model.predict(X_test)

y_pred_classes = np.argmax(y_pred, axis=1)

print(y_pred_classes)

pred = pd.concat([pd.DataFrame(y_test, columns=['y_test']), pd.DataFrame(y_pred_classes, columns=['y_pred'])], axis=1)
print(pred.head(10))

misclassified = pred[pred['y_test'] != pred['y_pred']]

plt.figure(figsize=(13,13))

for i, j in zip(range(1,12), misclassified.index[:10]):
    plt.subplot(1, 10, i)
    plt.axis('off')
    plt.imshow(X_test[j], cmap='gray_r')
    plt.title('y_test: ' + str(y_test[j]) + '\n' + 'y_pred: ' + str(y_pred_classes[j]), color='black', fontsize=12)
plt.show()

# Zapisanie modelu do pliku
model.save('mnist_model.h5')
