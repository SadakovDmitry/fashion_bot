import torch
import torchvision.models as models
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from aiogram import Bot, Dispatcher, types
from aiogram.types import ReplyKeyboardMarkup, KeyboardButton
from aiogram.utils import executor
from PIL import Image
import os

# Папки для хранения изображений
os.makedirs("images/top", exist_ok=True)
os.makedirs("images/bottom", exist_ok=True)

# Трансформации для изображения
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Изменяем размер для модели MobileNet
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Загрузка набора данных Fashion-MNIST
dataset = datasets.FashionMNIST(root='data', train=True, download=True, transform=transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Модель
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
model.classifier[3] = nn.Linear(1024, 10)  # 10 классов (Fashion-MNIST)
model.to(device)

# Оптимизатор
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Обучение
epochs = 10
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()

        # Прямой проход
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Обратное распространение
        loss.backward()
        optimizer.step()

        # Статистика
        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print(f"Эпоха {epoch + 1}/{epochs}, Потери: {running_loss/len(dataloader)}, Точность: {100 * correct / total}%")

# Сохраняем модель
torch.save(model.state_dict(), "fashion_mnist_model.pth")
