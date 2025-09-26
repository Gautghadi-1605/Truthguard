import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from torch.utils.data import DataLoader, Dataset
import os
from PIL import Image
from glob import glob
import random

# Define dataset path
dataset_path = "C:/finalyear/backend/fakedetection/Dataset/images"  # Update to your actual path

# Get all image paths from "real" and "fake" folders
real_images = glob(os.path.join(dataset_path, "Real", "*.jpg"))  
fake_images = glob(os.path.join(dataset_path, "Fake", "*.jpg"))

# Randomly select 64 images from each class
real_images = random.sample(real_images, min(len(real_images), 64))
fake_images = random.sample(fake_images, min(len(fake_images), 64))

# Combine selected images
image_paths = real_images + fake_images
labels = [0] * len(real_images) + [1] * len(fake_images)  # 0 = Real, 1 = Fake

# Define transformations
transform = transforms.Compose([
    transforms.Resize((128, 128)),  # Reduce image size for faster training
    transforms.ToTensor(),
])

# Dataset class
class ImageDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(self.labels[idx], dtype=torch.long)  # Convert labels properly

# Load dataset
dataset = ImageDataset(image_paths, labels, transform)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)  # Reduce batch size

# Load pre-trained ResNet18 (smaller than ResNet50, faster training)
model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
model.fc = nn.Linear(model.fc.in_features, 2)  # 2 classes: Real/Fake

# Define loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Train Model
num_epochs = 5
for epoch in range(num_epochs):
    running_loss = 0.0
    for images, labels in dataloader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels.clone().detach())  # Proper tensor handling
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    
    print(f"Epoch {epoch+1}/{num_epochs} - Loss: {running_loss/len(dataloader):.4f}")

# Save the model
os.makedirs("C:/finalyear/backend/fakedetection/models", exist_ok=True)
torch.save(model.state_dict(), "C:/finalyear/backend/fakedetection/models/image_fake_detector.pth")
print(" Image Fake Detector Model Saved!")


