Below are three progressively richer CNN examples for image classification. Each illustrates key concepts—from a small custom CNN on MNIST to a deeper network on CIFAR‑10, and finally transfer learning with a pretrained ResNet. Feel free to pick one (or all!) to deepen your CNN intuition.

## 1. Small CNN on MNIST
A classic “hello world” for vision: grayscale 28×28 handwritten digits.

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score

# 1. Data transforms & loaders
transform = transforms.Compose([
    transforms.ToTensor(),                           # → [0,1]
    transforms.Normalize((0.1307,), (0.3081,))       # zero‑mean, unit‑var
])
train_ds = datasets.MNIST(root=".", train=True,  download=True, transform=transform)
test_ds  = datasets.MNIST(root=".", train=False, download=True, transform=transform)
train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
test_loader  = DataLoader(test_ds,  batch_size=1000)

# 2. Define a small CNN via nn.Sequential
model = nn.Sequential(
    nn.Conv2d(1, 8, kernel_size=3, padding=1),      # → 8×28×28
    nn.ReLU(),
    nn.MaxPool2d(2),                                # → 8×14×14

    nn.Conv2d(8, 16, kernel_size=3, padding=1),     # → 16×14×14
    nn.ReLU(),
    nn.MaxPool2d(2),                                # → 16×7×7

    nn.Flatten(),                                   # → 16*7*7 = 784
    nn.Linear(784, 64),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(64, 10)                               # 10 classes
)

# 3. Loss, optimizer, device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 4. Training loop
for epoch in range(1, 6):
    model.train()
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        loss = criterion(logits, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch} complete")

# 5. Evaluation
model.eval()
all_preds, all_labels = [], []
with torch.no_grad():
    for xb, yb in test_loader:
        xb = xb.to(device)
        preds = model(xb).argmax(dim=1).cpu()
        all_preds.append(preds)
        all_labels.append(yb)
acc = accuracy_score(torch.cat(all_labels), torch.cat(all_preds))
print(f"MNIST Test Accuracy: {acc:.3f}")
```
What to explore next:

- Increase filter counts (e.g. 32 → 64) or add a third conv block.

- Plot training & validation loss curves.

- Experiment with different pooling (AvgPool) or activation (LeakyReLU).

## 2. Deeper CNN on CIFAR‑10
Color images (32×32×3), 10 classes. Shows data augmentation.

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score

# 1. Transforms with augmentation
train_tf = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize((0.4914,0.4822,0.4465), (0.2023,0.1994,0.2010))
])
test_tf = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914,0.4822,0.4465), (0.2023,0.1994,0.2010))
])

train_ds = datasets.CIFAR10(root=".", train=True,  download=True, transform=train_tf)
test_ds  = datasets.CIFAR10(root=".", train=False, download=True, transform=test_tf)
train_loader = DataLoader(train_ds, batch_size=128, shuffle=True, num_workers=2)
test_loader  = DataLoader(test_ds,  batch_size=256, shuffle=False, num_workers=2)

# 2. Define a deeper CNN
model = nn.Sequential(
    # Block 1
    nn.Conv2d(3, 32, kernel_size=3, padding=1),
    nn.BatchNorm2d(32),
    nn.ReLU(),
    nn.Conv2d(32, 32, kernel_size=3, padding=1),
    nn.BatchNorm2d(32),
    nn.ReLU(),
    nn.MaxPool2d(2),              # → 32×16×16

    # Block 2
    nn.Conv2d(32, 64, kernel_size=3, padding=1),
    nn.BatchNorm2d(64),
    nn.ReLU(),
    nn.Conv2d(64, 64, kernel_size=3, padding=1),
    nn.BatchNorm2d(64),
    nn.ReLU(),
    nn.MaxPool2d(2),              # → 64×8×8

    nn.Flatten(),                 # → 64*8*8 = 4096
    nn.Linear(4096, 256),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(256, 10)
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

# 3. Training
for epoch in range(1, 51):
    model.train()
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        loss = criterion(model(xb), yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    scheduler.step()

    if epoch % 10 == 0:
        # quick eval
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for xb, yb in test_loader:
                xb = xb.to(device)
                preds = model(xb).argmax(dim=1).cpu()
                all_preds.append(preds)
                all_labels.append(yb)
        acc = accuracy_score(torch.cat(all_labels), torch.cat(all_preds))
        print(f"Epoch {epoch:02d} — Test Acc: {acc:.3f}")
```
What to explore next:

- Try different schedulers (CosineAnnealingLR, OneCycleLR).

- Visualize feature maps by hooking into conv layers.

- Compare performance with/without batch norm or dropout.

### 3. Transfer Learning with Pretrained ResNet on a Custom Dataset
Fine‑tune a ResNet‑18 for your own 5‑class image folder.

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score

# 1. Data loaders (assumes folder structure train/<class>/*.jpg, val/<class>/*.jpg)
data_tf = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
    ])
}
train_ds = datasets.ImageFolder("data/train", transform=data_tf['train'])
val_ds   = datasets.ImageFolder("data/val",   transform=data_tf['val'])
train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=4)
val_loader   = DataLoader(val_ds,   batch_size=64, shuffle=False, num_workers=4)

# 2. Load pretrained ResNet‑18 and replace final layer
model = models.resnet18(pretrained=True)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 5)   # your 5 classes

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 3. Freeze early layers (optional)
for name, param in model.named_parameters():
    if "layer4" not in name and "fc" not in name:
        param.requires_grad = False

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

# 4. Training & validation
for epoch in range(1, 16):
    model.train()
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        loss = criterion(model(xb), yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    scheduler.step()

    # Evaluate
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for xb, yb in val_loader:
            xb = xb.to(device)
            preds = model(xb).argmax(dim=1).cpu()
            all_preds.append(preds)
            all_labels.append(yb)
    acc = accuracy_score(torch.cat(all_labels), torch.cat(all_preds))
    print(f"Epoch {epoch:02d} — Val Acc: {acc:.3f}")
```
What to explore next:

- Unfreeze more layers after a few epochs for deeper fine‑tuning.

- Try larger architectures (ResNet‑50, EfficientNet).

- Integrate early stopping or mixed‑precision (with torch.cuda.amp).

### Tips for All Examples
- Move tensors & model to GPU (shown above) for speed.

- Log training/validation metrics and plot them to detect overfitting.

- Experiment: number of filters, kernel sizes, depths, learning rates, augmentations.

- Visualize learned filters or activation maps to build intuition.

Pick the example that matches your next learning goal, and let me know if you’d like to dive deeper into any step!
