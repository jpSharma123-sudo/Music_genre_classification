import librosa
import numpy as np
import os

def extract_features(directory, genres, max_len=130):
    data = []
    labels = []

    for genre in genres:
        genre_dir = os.path.join(directory, genre)
        for filename in os.listdir(genre_dir):
            file_path = os.path.join(genre_dir, filename)
            y, sr = librosa.load(file_path, duration=30)
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
            if mfcc.shape[1] < max_len:
                pad = max_len - mfcc.shape[1]
                mfcc = np.pad(mfcc, ((0, 0), (0, pad)), mode='constant')
            else:
                mfcc = mfcc[:, :max_len]
            data.append(mfcc)
            labels.append(genres.index(genre))

    return np.array(data), np.array(labels)
