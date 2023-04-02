from flask import Flask, render_template, request
import pandas as pd
from tensorflow.keras.models import load_model
import joblib

app = Flask(__name__)

model = load_model('trained_model.h5')
scaler = joblib.load('scaler.pkl')

@app.route('/')
def index():
    return render_template('california_housing_index.html')

@app.route('/predict', methods=['POST'])
def predict():
    new_property = {
        'MedInc': float(request.form['MedInc']),
        'HouseAge': float(request.form['HouseAge']),
        'AveRooms': float(request.form['AveRooms']),
        'AveBedrms': float(request.form['AveBedrms']),
        'Population': float(request.form['Population']),
        'AveOccup': float(request.form['AveOccup']),
        'Latitude': float(request.form['Latitude']),
        'Longitude': float(request.form['Longitude'])
    }

    new_property_df = pd.DataFrame([new_property])
    scaled_new_property = scaler.transform(new_property_df)
    predicted_price = model.predict(scaled_new_property)
    property_value = predicted_price[0][0] * 100000
    formatted_value = format(property_value, ',.0f').replace(',', '.')

    return render_template('california_housing_result.html', price=formatted_value)

    #print(f'Prognozowana cena nieruchomości: {formatted_value} dolarów.')


if __name__ == '__main__':
    app.run(port=8080, debug=True)