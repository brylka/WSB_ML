import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pickle

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Wczytaj dane (tutaj należy użyć własnego zbioru danych z cechami dźwiękowymi i etykietami gatunków)
data = pd.read_csv("music_features.csv")

# Przygotuj dane
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

# Koduj etykiety gatunków
encoder = LabelEncoder()
y = encoder.fit_transform(y)

# Podziel dane na treningowe i testowe
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Skaluj dane
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Stwórz model
model = Sequential()
model.add(Dense(units=128, activation='relu', input_shape=(X_train.shape[1],)))
model.add(Dense(units=64, activation='relu'))
model.add(Dense(units=np.unique(y).size, activation='softmax'))

# Kompiluj model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Trenuj model
history = model.fit(X_train, y_train, epochs=100, validation_split=0.1)

# Ewaluuj model na danych testowych
model.evaluate(X_test, y_test)

# Zapisywanie modelu do pliku
model.save('music.h5')

# Zapisz skalownik do pliku
with open('music_scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

# Zapisz LabelEncoder do pliku
with open('music_encoder.pkl', 'wb') as f:
    pickle.dump(encoder, f)