import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

# =======================
# 1. TRANSFORM ẢNH
# =======================

train_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

val_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# =======================
# 2. LOAD DATA
# =======================

train_dir = r"C:\Users\Asus\OneDrive\Desktop\GIT\Fruit_Classification\Dataset\clean\train"
val_dir   = r"C:\Users\Asus\OneDrive\Desktop\GIT\Fruit_Classification\Dataset\clean\val"

train_data = datasets.ImageFolder(train_dir, transform=train_tf)
val_data   = datasets.ImageFolder(val_dir, transform=val_tf)

train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
val_loader   = DataLoader(val_data, batch_size=16, shuffle=False)

# =======================
# 3. MODEL
# =======================

model = models.resnet18(pretrained=True)
model.fc = nn.Linear(512, len(train_data.classes))   # số class tự động

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)

# =======================
# 4. OPTIMIZER
# =======================

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# =======================
# 5. TRAIN LOOP
# =======================

EPOCHS = 10

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0

    for img, label in train_loader:
        img, label = img.to(device), label.to(device)

        optimizer.zero_grad()
        pred = model(img)
        loss = criterion(pred, label)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {total_loss:.4f}")

# =======================
# 6. LƯU MODEL
# =======================

torch.save(model.state_dict(), "fruit_model.pth")

print("Đã train xong và lưu model!")
