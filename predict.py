import librosa
import numpy as np
from keras.models import load_model

GENRES = ["blues", "classical", "country", "disco", "hiphop",
          "jazz", "metal", "pop", "reggae", "rock"]

def predict(audio_path, model_path="model/cntn_model.h5"):
    # Load the trained model
    model = load_model(model_path)
    if model is None:
        raise ValueError(f"Failed to load model from {model_path}")

    # Load the audio file
    y, sr = librosa.load(audio_path, duration=30)

    # Extract MFCC features
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)

    # Normalize length to 130 frames
    max_len = 130
    if mfcc.shape[1] < max_len:
        pad_width = max_len - mfcc.shape[1]
        mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')
    else:
        mfcc = mfcc[:, :max_len]

    # Reshape to match model input shape: (1, 40, 130, 1)
    X = np.expand_dims(mfcc, axis=(0, -1))  # shape becomes (1, 40, 130, 1)

    # Make prediction
    prediction = model.predict(X, verbose=0)[0]
    predicted_index = np.argmax(prediction)
    predicted_genre = GENRES[predicted_index]
    confidence = float(prediction[predicted_index])

    return predicted_genre, confidence

# Example use
if __name__ == "__main__":
    genre, confidence = predict("data/genres/blues/blues_sample.wav")
    print(f"Predicted Genre: {genre} ({confidence:.2f} confidence)")
