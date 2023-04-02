import pandas as pd
from tensorflow.keras.models import load_model
import joblib

# Wczytanie modelu z dysku
model = load_model('trained_model.h5')

# Wczytanie skalera z dysku
scaler = joblib.load('scaler.pkl')

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

new_property_df = pd.DataFrame([new_property])

scaled_new_property = scaler.transform(new_property_df)

predicted_price = model.predict(scaled_new_property)

property_value = predicted_price[0][0] * 100000
formatted_value = format(property_value, ',.0f').replace(',', '.')
print(f'Prognozowana cena nieruchomości: {formatted_value} dolarów.')