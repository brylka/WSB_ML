import os
import librosa
import pandas as pd
import numpy as np

genres = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']
data = []

for genre in genres:
    for filename in os.listdir(f'./genres/{genre}'):
        file_path = f'./genres/{genre}/{filename}'
        y, sr = librosa.load(file_path, duration=30)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13).T
        chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr).T
        contrast = librosa.feature.spectral_contrast(y=y, sr=sr).T

        for i in range(mfcc.shape[0]):
            row = np.hstack([mfcc[i], chroma_stft[i], contrast[i], genre])
            data.append(row)

data = np.array(data)
data_df = pd.DataFrame(data, columns=['mfcc_' + str(i) for i in range(13)] +
                                     ['chroma_stft_' + str(i) for i in range(12)] +
                                     ['contrast_' + str(i) for i in range(7)] + ['genre'])
data_df.to_csv('music_features.csv', index=False)