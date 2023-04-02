import numpy as np
import cv2
from PIL import Image
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Wczytanie wytrenowanego modelu
model = load_model('mnist_model.h5')

def process_image(image):
    resized_image = cv2.resize(image, (28, 28), interpolation=cv2.INTER_AREA)
    inverted_image = cv2.bitwise_not(resized_image)
    scaled_image = inverted_image / 255.0
    input_image = np.expand_dims(scaled_image, axis=0)
    input_image = input_image.reshape(1, 28, 28)
    integer_image = np.round(input_image * 255).astype(np.uint8)
    return integer_image

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Wczytanie obrazu z przesłanego pliku
        image_file = request.files['image']
        image = Image.open(image_file).convert('L')
        image = np.array(image)

        # Przetwarzanie obrazu i wykonanie predykcji
        input_image = process_image(image)
        prediction = model.predict(input_image, verbose=0)
        predicted_digit = np.argmax(prediction)

        return render_template('clasyfication_server.html', prediction=predicted_digit)
    return render_template('clasyfication_server.html')


if __name__ == '__main__':
    app.run(port=8080, debug=True)