import numpy as np
import cv2
from tensorflow.keras.models import load_model


# Wczytanie modelu
model = load_model('mnist_model.h5')

for i in range(10):

    # Wczytanie obrazu cyfry z pliku .jpg
    image_file = 'digits/digit_'+str(i)+'.jpg'
    image = cv2.imread(image_file, cv2.IMREAD_GRAYSCALE)

    # Przetworzenie obrazu
    resized_image = cv2.resize(image, (28, 28), interpolation=cv2.INTER_AREA)
    inverted_image = cv2.bitwise_not(resized_image)
    scaled_image = inverted_image / 255.0
    input_image = np.expand_dims(scaled_image, axis=0)
    input_image = input_image.reshape(1, 28, 28)
    integer_image = np.round(input_image * 255).astype(np.uint8)

    # Wykonanie predykcji
    prediction = model.predict(integer_image, verbose=0)
    predicted_digit = np.argmax(prediction)

    print(f'Wczytana cyfra: {i} Przewidziana cyfra: {predicted_digit}')