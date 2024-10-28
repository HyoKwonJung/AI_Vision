import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from CustomDataset import CustomDataset
from torchvision.models import ResNet18_Weights

# 데이터 라벨링 함수 / 이미지의 이름, 소속되어 있는 폴더, 라벨링(0~)
# 폴더명은 "국적_선종_상태" 형태로 이루어져 있음.
def datalabeling(dataset_dir):
    image_data = []
    for class_label, class_folder in enumerate(os.listdir(dataset_dir)):
        class_path = os.path.join(dataset_dir, class_folder)
        if os.path.isdir(class_path):
            for image_file in os.listdir(class_path):
                image_path = os.path.join(class_path, image_file)
                if os.path.isfile(image_path):
                    image_data.append([image_file, class_folder, class_label])
    df = pd.DataFrame(image_data, columns=['Image Name', 'Class Folder', 'Class Label'])
    return df

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


trainDataset_dir = './trainDataset'
testDataset_dir = './testDataset'
training_data_label = datalabeling(trainDataset_dir)
test_data_label = datalabeling(testDataset_dir)

mean = [0.32546, 0.35472, 0.48292]
std = [0.21112, 0.20426, 0.24468]


train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    # transforms.RandomVerticalFlip(),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.3),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize(torch.Tensor(mean), torch.Tensor(std))
])
test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(torch.Tensor(mean), torch.Tensor(std))
])

# 데이터 로드와 트레이닝/확인용 데이터로 구분
trainingData = CustomDataset(training_data_label, trainDataset_dir, transform=train_transform)
train_size = int(0.90 * len(trainingData)) 
val_size = len(trainingData) - train_size
trainData, valData = random_split(trainingData, [train_size, val_size])

trainLoader = DataLoader(trainData, batch_size=4, shuffle=True)
valLoader = DataLoader(valData, batch_size=4, shuffle=False)
#테스트용 데이터 로드
testData = CustomDataset(test_data_label, testDataset_dir, transform=test_transform)
testLoader = DataLoader(testData, batch_size=4, shuffle=False)




#model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
# num_classes = len(training_data_label['Class Label'].unique())
# model.fc = nn.Linear(model.fc.in_features, num_classes)
# model = model.to(device)

# DenseNet을 사용하여 학습
model = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
num_classes = len(training_data_label['Class Label'].unique())
model.classifier = nn.Linear(model.classifier.in_features, num_classes)
model = model.to(device)

# 손실함수와 옵티마이저
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0002, weight_decay=0.0001)

# 정확도 계산함수
def calculate_accuracy(loader, model):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * (correct / total)
    return accuracy

# 에폭설정
num_epochs = 20
best_val_accuracy = 0.0

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for batch_idx, (images, labels) in enumerate(trainLoader):
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if (batch_idx + 1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{batch_idx+1}/{len(trainLoader)}], Loss: {loss.item():.4f}')

    epoch_loss = running_loss / len(trainLoader)
    val_accuracy = calculate_accuracy(valLoader, model)
    print(f'Epoch [{epoch+1}/{num_epochs}] Loss: {epoch_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%')

    # Save the best model based on validation accuracy
    if val_accuracy > best_val_accuracy:
        torch.save(model.state_dict(), 'best_model_dens.pth')
        best_val_accuracy = val_accuracy

# Load the best model after training
model.load_state_dict(torch.load('best_model_dens.pth'))

# Validate the best model on the test dataset
test_accuracy = calculate_accuracy(testLoader, model)
print(f'Best model test accuracy: {test_accuracy:.2f}%')
