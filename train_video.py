import os
import cv2  # type: ignore
import numpy as np  # type: ignore
import tensorflow as tf  # type: ignore
from tensorflow.keras.models import Sequential  # type: ignore
from tensorflow.keras.layers import Conv3D, LSTM, Dense, MaxPooling3D, BatchNormalization, Reshape  # type: ignore

# 📌 Step 1: Dataset Paths
dataset_path = os.path.join("backend", "fakedetection", "Dataset", "videos")
real_path = os.path.join(dataset_path, "DFD_original sequences")
fake_path = os.path.join(dataset_path, "DFD_manipulated_sequences", "DFD_manipulated_sequences")

# ✅ Ensure dataset folders exist
if not os.path.exists(real_path) or not os.path.exists(fake_path):
    print(" Dataset folders not found!")
    exit()

# 📌 Step 2: Extract Frames (5 Frames per Video)
def extract_frames(video_path, max_frames=5):
    cap = cv2.VideoCapture(video_path)
    frames = []
    
    while len(frames) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (64, 64))  # Resize to match model input
        frames.append(frame)

    cap.release()

    # ✅ Pad with black frames if video has fewer than max_frames
    while len(frames) < max_frames:
        frames.append(np.zeros((64, 64, 3), dtype=np.uint8))  

    return np.array(frames, dtype=np.float32) / 255.0  # Normalize pixel values

# 📌 Step 3: Load Dataset (Limit for Speed)
real_videos = [os.path.join(real_path, f) for f in os.listdir(real_path) if f.endswith(".mp4")][:300]
fake_videos = [os.path.join(fake_path, f) for f in os.listdir(fake_path) if f.endswith(".mp4")][:300]

video_files = real_videos + fake_videos
labels = np.array([0] * len(real_videos) + [1] * len(fake_videos))  # 0 = Real, 1 = Fake

# ✅ Check if videos exist
if len(video_files) == 0:
    print(" No videos found!")
    exit()

print(f" Found {len(real_videos)} real and {len(fake_videos)} fake videos.")

# 📌 Step 4: Data Generator (Efficient Batch Loading)
def data_generator(video_files, labels, batch_size=16):
    while True:
        indices = np.random.permutation(len(video_files))  # Shuffle dataset each epoch
        for i in range(0, len(video_files), batch_size):
            batch_indices = indices[i:i + batch_size]
            batch_videos = [video_files[idx] for idx in batch_indices]
            batch_labels = [labels[idx] for idx in batch_indices]
            batch_frames = [extract_frames(f) for f in batch_videos]
            yield np.array(batch_frames), np.array(batch_labels)

train_generator = data_generator(video_files, labels, batch_size=16)

# 📌 Step 5: Define CNN + LSTM Model
model = Sequential([
    Conv3D(16, (3, 3, 3), padding="same", input_shape=(5, 64, 64, 3)),
    BatchNormalization(),
    tf.keras.layers.ReLU(),  # ✅ Activation after batch normalization
    MaxPooling3D(pool_size=(1, 2, 2)),

    Reshape((5, -1)),  # ✅ Preserve Time Dimension for LSTM

    LSTM(16, return_sequences=False),
    Dense(16, activation="relu"),
    Dense(1, activation="sigmoid")  # Binary classification
])

model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

# 📌 Step 6: Train Model
print(" Training model...")
model.fit(train_generator, steps_per_epoch=len(video_files) // 16, epochs=5)

# 📌 Step 7: Save Model in backend/fakedetection/models
save_dir = os.path.join("backend", "fakedetection", "models")
os.makedirs(save_dir, exist_ok=True)  # ✅ Ensure the directory exists

save_path = os.path.join(save_dir, "video_fake_detector.keras")
model.save(save_path)
print(f" Model saved at {save_path}")

