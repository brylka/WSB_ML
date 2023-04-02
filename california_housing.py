import pandas as pd
import matplotlib.pyplot as plt

# Importowanie potrzebnych funkcji z bibliotek sklearn i tensorflow
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Pobieranie zestawu danych dotyczących cen mieszkań w Kalifornii
california_housing = fetch_california_housing()
# Przypisanie danych do zmiennych X (cechy) i y (wartości docelowe)
X, y = california_housing.data, california_housing.target

# Podział danych na zestawy treningowe i testowe
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Inicjalizacja StandardScaler do skalowania danych
scaler = StandardScaler()
# Skalowanie danych treningowych
X_train = scaler.fit_transform(X_train)
# Skalowanie danych testowych
X_test = scaler.transform(X_test)

# Tworzenie modelu sekwencyjnego przy użyciu TensorFlow
model = Sequential()
# Dodanie warstwy Dense z 64 jednostkami i funkcją aktywacji ReLU
model.add(Dense(units=64, activation='relu', input_shape=(X_train.shape[1],)))
# Dodanie kolejnej warstwy Dense z 32 jednostkami i funkcją aktywacji ReLU
model.add(Dense(units=32, activation='relu'))
# Dodanie warstwy wyjściowej z 1 jednostką
model.add(Dense(units=1))

# Kompilacja modelu z optymalizatorem Adam i funkcją straty mean_squared_error
model.compile(optimizer='adam', loss='mean_squared_error')

# Wyświetlanie podsumowania modelu
model.summary()

# Trenowanie modelu z 100 epokami i walidacją na 10% danych treningowych
history = model.fit(X_train, y_train, epochs=100, validation_split=0.1)

# Wykres funkcji straty dla danych treningowych i walidacyjnych
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Przewidywanie wartości dla danych testowych
y_pred = model.predict(X_test)
# Obliczanie błędu średniokwadratowego (MSE) dla danych testowych
mse = tf.keras.losses.mean_squared_error(y_test, y_pred)
print(f'Mean squared error: {mse.numpy().mean()}')

# Nowa nieruchomość, dla której chcemy oszacować cenę
new_property = {
    'MedInc': 8.32,
    'HouseAge': 25,
    'AveRooms': 6.23,
    'AveBedrms': 1.01,
    'Population': 1800,
    'AveOccup': 3.5,
    'Latitude': 37.88,
    'Longitude': -122.23
}

# Tworzenie DataFrame z nowymi danymi
new_property_df = pd.DataFrame([new_property])

# Skalowanie danych wejściowych przy użyciu tego samego skalera użytego do trenowania modelu
scaled_new_property = scaler.transform(new_property_df)

# Prognozowanie ceny przy użyciu wytrenowanego modelu
predicted_price = model.predict(scaled_new_property)

# Wypisanie prognozowanej ceny
print(f'Prognozowana cena nieruchomości: {predicted_price[0][0]:.3f} (w jednostkach 100,000 USD)')