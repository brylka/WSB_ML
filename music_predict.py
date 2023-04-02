import numpy as np
import pandas as pd
import librosa
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model
import pickle

def load_encoder(encoder_path):
    with open(encoder_path, 'rb') as f:
        return pickle.load(f)

def load_scaler(scaler_path):
    with open(scaler_path, 'rb') as f:
        return pickle.load(f)

def extract_features(file_path):
    y, sr = librosa.load(file_path, duration=30)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13).T
    chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr).T
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr).T

    features = []
    for i in range(mfcc.shape[0]):
        row = np.hstack([mfcc[i], chroma_stft[i], contrast[i]])
        features.append(row)

    return pd.DataFrame(features)

def predict_genre(file_path, model_path, scaler_path, encoder_path):
    # Wczytaj wytrenowany model
    model = load_model(model_path)

    # Wczytaj wytrenowany skalownik
    scaler = load_scaler(scaler_path)

    # Wczytaj LabelEncoder
    encoder = load_encoder(encoder_path)

    # Wyodrębnij cechy dźwiękowe z pliku muzycznego
    new_song = extract_features(file_path)

    # Skaluj cechy dźwiękowe przy użyciu wcześniej wytrenowanego skalera
    scaled_new_song = scaler.transform(new_song)

    # Przewiduj gatunek muzyki
    predicted_prob = model.predict(scaled_new_song)
    predicted_genre = np.argmax(np.mean(predicted_prob, axis=0))

    # Kodowanie etykiet gatunków
    genres = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']

    # Dekoduj przewidywane etykiety gatunków
    return encoder.inverse_transform([predicted_genre])[0]

file_path = 'mp3/reggae3.mp3'
model_path = 'music.h5'
scaler_path = 'music_scaler.pkl'
encoder_path = 'music_encoder.pkl'

predicted_genre = predict_genre(file_path, model_path, scaler_path, encoder_path)
print(f'Prognozowany gatunek muzyki: {predicted_genre}')