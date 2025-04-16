import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import time
import os

# 安裝pip install torch torchvision


# 設定設備
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 下載並處理數據
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # MobileNetV3 預設的輸入尺寸是 224x224
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 用 ImageNet 預訓練模型的標準化
])

train_dataset = datasets.ImageFolder(root='./datasets/train', transform=transform)
val_dataset = datasets.ImageFolder(root='./datasets/valid', transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# 載入 MobileNetV3 預訓練模型
model = torchvision.models.mobilenet_v3_small(pretrained=True)  # 使用預訓練模型
num_ftrs = model.classifier[3].in_features  # 取得最後一層的輸入特徵數
model.classifier[3] = nn.Linear(num_ftrs, len(train_dataset.classes))  # 修改為我們自己的類別數

# 將模型轉移到適當的設備（GPU 或 CPU）
model = model.to(device)

# 設定損失函數和優化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 訓練模型
num_epochs = 300
best_val_accuracy = 0.0

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    # 訓練迴圈
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        
        # 前向傳播
        outputs = model(images)
        loss = criterion(outputs, labels)

        # 反向傳播和優化
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        # 計算正確率
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    # 計算訓練集上的正確率
    train_accuracy = 100 * correct / total
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}, Accuracy: {train_accuracy:.2f}%")

    # 驗證模型
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_accuracy = 100 * correct / total
    print(f"Validation Accuracy: {val_accuracy:.2f}%")

    # 保存最佳模型
    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        print(f"Saving model with accuracy {best_val_accuracy:.2f}%")
        torch.save(model.state_dict(), "best_mobilenetv3.pth")  # 儲存最佳模型

print("訓練完成！")
