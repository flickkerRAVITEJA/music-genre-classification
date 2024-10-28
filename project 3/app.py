from flask import Flask, request, jsonify
import librosa
import numpy as np
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model("genre_classifier.h5")

app = Flask(__name__)

def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=22050)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13).mean(axis=1)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr).mean(axis=1)
    mel = librosa.feature.melspectrogram(y=y, sr=sr).mean(axis=1)
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)

    return np.hstack([mfcc, chroma, mel, tempo])

@app.route('/predict', methods=['POST'])
def predict_genre():
    return "Predection Route"
    file = request.files['file']
    features = extract_features(file)
    prediction = model.predict(features.reshape(1, -1)).argmax()
    genres = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']
    return jsonify({'genre': genres[prediction]})

if __name__ == '__main__':
    app.run(debug=True)
