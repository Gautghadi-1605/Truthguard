import librosa
import numpy as np
import os
import tensorflow as tf # type: ignore
import random

# Define dataset paths
DATASET_PATH = r"C:\finalyear\backend\fakedetection\Dataset\audio"
REAL_PATH = os.path.join(DATASET_PATH, "real")
FAKE_PATH = os.path.join(DATASET_PATH, "fake")

# Function to load and extract MFCC features
def extract_features(folder, label, max_files=50):  # Limit files to speed up training
    audio_data, labels = [], []
    files = [f for f in os.listdir(folder) if f.endswith(".wav") or f.endswith(".mp3")]
    random.shuffle(files)  # Shuffle to ensure variety
    files = files[:max_files]  # Limit number of files
    
    for file in files:
        file_path = os.path.join(folder, file)
        y, sr = librosa.load(file_path, sr=16000)  # Lower sample rate
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=10)  # Reduce MFCC size
        audio_data.append(np.mean(mfcc, axis=1))
        labels.append(label)

    return audio_data, labels

# Load dataset (reduce number of samples)
real_audio, real_labels = extract_features(REAL_PATH, 0, max_files=50)
fake_audio, fake_labels = extract_features(FAKE_PATH, 1, max_files=50)

# Combine real and fake audio
X = np.array(real_audio + fake_audio)
y = np.array(real_labels + fake_labels)

# Build Model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation="relu", input_shape=(10,)),  # Adjusted for 10 MFCCs
    tf.keras.layers.Dense(32, activation="relu"),
    tf.keras.layers.Dense(1, activation="sigmoid")
])

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
model.fit(X, y, epochs=5, validation_split=0.2, batch_size=8)  # Fewer epochs & smaller batch size

# Save the model
os.makedirs("fakedetection/models", exist_ok=True)  # Ensure directory exists
model.save(r"C:\finalyear\backend\fakedetection\models\audio_fake_news_model.h5")
print("Audio Fake Detector Model Saved!")
