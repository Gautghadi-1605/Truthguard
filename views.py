import json
import os
import librosa
import numpy as np
import torch
import pickle
import tensorflow as tf
from PIL import Image
import cv2
from uuid import uuid4

from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from torchvision import models, transforms

# === Model Paths ===
BASE_PATH = r'C:\finalyear\backend\fakedetection\models'
AUDIO_MODEL_PATH = os.path.join(BASE_PATH, 'audio_fake_news_model.h5')
IMAGE_MODEL_PATH = os.path.join(BASE_PATH, 'image_fake_detector.pth')
VIDEO_MODEL_PATH = os.path.join(BASE_PATH, 'video_fake_detector.keras')
TFIDF_PATH = r'C:\finalyear\backend\fakedetection\tfidf_vectori.pkl'
TEXT_MODEL_PATH = os.path.join(BASE_PATH, 'passive_aggressive_mo.pkl')

# === Load Models ===
try:
    print("📦 Loading models...")

    audio_model = tf.keras.models.load_model(AUDIO_MODEL_PATH)
    print("✅ Audio model loaded.")

    image_model = models.resnet18()
    image_model.fc = torch.nn.Linear(image_model.fc.in_features, 2)
    image_model.load_state_dict(torch.load(IMAGE_MODEL_PATH, map_location=torch.device('cpu')))
    image_model.eval()
    print("✅ Image model loaded.")

    video_model = tf.keras.models.load_model(VIDEO_MODEL_PATH)
    print("✅ Video model loaded.")

    with open(TFIDF_PATH, 'rb') as f:
        tfidf_vectorizer = pickle.load(f)
    with open(TEXT_MODEL_PATH, 'rb') as f:
        text_model = pickle.load(f)
    print("✅ Text model and vectorizer loaded.")

except Exception as e:
    print(f"❌ Error loading models: {e}")

# === Predict Functions ===
def predict_audio(file_path):
    try:
        y, sr = librosa.load(file_path, sr=16000)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=10)
        mfcc = np.mean(mfcc, axis=1).reshape(1, -1)
        prediction = audio_model.predict(mfcc)
        return 'Fake' if prediction[0] > 0.5 else 'Real'
    except Exception as e:
        return f"Audio processing error: {str(e)}"

def predict_image(image_path):
    try:
        transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor()
        ])
        image = Image.open(image_path).convert("RGB")
        image = transform(image).unsqueeze(0)
        with torch.no_grad():
            output = image_model(image)
            _, predicted = torch.max(output, 1)
        return 'Fake' if predicted.item() == 1 else 'Real'
    except Exception as e:
        return f"Image processing error: {str(e)}"

def predict_video(video_path):
    try:
        cap = cv2.VideoCapture(video_path)
        frames = []
        while len(frames) < 5:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, (64, 64))
            frames.append(frame)
        while len(frames) < 5:
            frames.append(np.zeros((64, 64, 3), dtype=np.uint8))
        frames = np.array(frames) / 255.0
        prediction = video_model.predict(np.expand_dims(frames, axis=0))
        return 'Fake' if prediction[0][0] > 0.5 else 'Real'
    except Exception as e:
        return f"Video processing error: {str(e)}"

def predict_text(text):
    try:
        X_tfidf = tfidf_vectorizer.transform([text])
        prediction = text_model.predict(X_tfidf)
        return 'Fake' if prediction[0] == 1 else 'Real'
    except Exception as e:
        return f"Text processing error: {str(e)}"

# === Main API Endpoint ===
@csrf_exempt
def detect_fake(request):
    print("📨 detect_fake called")
    print("🔍 Request method:", request.method)
    print("🧾 Content type:", request.content_type)

    if request.method != 'POST':
        return JsonResponse({'error': 'Invalid request method'}, status=400)

    try:
        # Debug: Print raw request data
        print("📨 Raw request.POST data:", request.POST)
        print("📨 Raw request.FILES data:", request.FILES)

        # === JSON (Text Only)
        if request.content_type.startswith('application/json'):
            try:
                data = json.loads(request.body.decode('utf-8'))
            except json.JSONDecodeError:
                return JsonResponse({'error': 'Invalid JSON format'}, status=400)

            file_type = data.get('file_type')
            text = data.get('text')

            print("📨 Received file_type (json):", file_type)
            print("📝 Text:", text)

            if file_type != 'text':
                return JsonResponse({'error': 'Invalid file_type for JSON. Use \"text\".'}, status=400)

            if not text:
                return JsonResponse({'error': 'Missing text data'}, status=400)

            result = predict_text(text)
            return JsonResponse({'result': result})

        # === Multipart (image, audio, video)
        elif request.content_type.startswith('multipart/form-data'):
            file_type = request.POST.get('file_type')
            uploaded_file = request.FILES.get('file')

            print("📨 Received file_type (multipart):", file_type)
            print("📁 Uploaded file:", uploaded_file)

            if not file_type or not uploaded_file:
                return JsonResponse({'error': 'Missing file_type or file'}, status=400)

            # Save file temporarily
            temp_dir = 'temp'
            os.makedirs(temp_dir, exist_ok=True)
            ext = os.path.splitext(uploaded_file.name)[-1]
            temp_filename = f"{uuid4().hex}{ext}"
            temp_path = os.path.join(temp_dir, temp_filename)

            with open(temp_path, 'wb+') as f:
                for chunk in uploaded_file.chunks():
                    f.write(chunk)

            print(f"📦 File saved to: {temp_path}")

            if file_type.lower() == 'image':
                result = predict_image(temp_path)
            elif file_type.lower() == 'audio':
                result = predict_audio(temp_path)
            elif file_type.lower() == 'video':
                result = predict_video(temp_path)
            else:
                os.remove(temp_path)
                return JsonResponse({'error': 'Unsupported file_type. Use image, audio, video or text.'}, status=400)

            os.remove(temp_path)
            return JsonResponse({'result': result})

        else:
            return JsonResponse({'error': 'Unsupported content type'}, status=400)

    except Exception as e:
        print(f"❌ Exception in detect_fake: {e}")
        return JsonResponse({'error': str(e)}, status=500)

# === Basic Test View ===
def home(request):
    return JsonResponse({'message': 'Welcome to Fake News Detection API!'})



