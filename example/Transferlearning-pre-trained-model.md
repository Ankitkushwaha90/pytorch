Below are three in‑depth Transfer‑Learning examples—each building on the last—to give you a deep, practical grasp of how to leverage pretrained vision models in PyTorch. You’ll see both feature‑extraction and fine‑tuning strategies, as well as how to swap in different backbones.

### Example 1: Feature‑Extraction with ResNet18
Freeze the entire backbone, train only a new classifier head.

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, datasets, transforms
from torch.utils.data import DataLoader

# 1. Data transforms & loaders
tf = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])
train_ds = datasets.ImageFolder("data/train", transform=tf)
val_ds   = datasets.ImageFolder("data/val",   transform=tf)
train_loader = DataLoader(train_ds, batch_size=32, shuffle=True,  num_workers=4)
val_loader   = DataLoader(val_ds,   batch_size=64, shuffle=False, num_workers=4)

# 2. Load pretrained ResNet18
model = models.resnet18(pretrained=True)

# 3. Freeze all layers
for param in model.parameters():
    param.requires_grad = False

# 4. Replace the final FC layer
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, len(train_ds.classes))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 5. Train only the head
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=1e-3)

for epoch in range(1, 6):
    model.train()
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        loss = criterion(logits, yb)
        optimizer.zero_grad(); loss.backward(); optimizer.step()
    print(f"Epoch {epoch} complete")
```
Why it matters:

- Fast & cheap since only ~0.5 M parameters (the head) are updated.

- Great when you have limited data.

### Example 2: Fine‑Tuning Later Layers (Gradual Unfreezing)
Freeze early layers; unfreeze deeper layers for more adaptable features.

```python
# Starting from the model in Example 1

# 1. Freeze everything except layer4 and FC
for name, param in model.named_parameters():
    if "layer4" in name or "fc" in name:
        param.requires_grad = True
    else:
        param.requires_grad = False

# 2. Optimizer on trainable params only
trainable = [p for p in model.parameters() if p.requires_grad]
optimizer = optim.SGD(trainable, lr=1e-3, momentum=0.9)

# 3. Training loop with LR scheduler
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
for epoch in range(1, 11):
    model.train()
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        loss = criterion(model(xb), yb)
        optimizer.zero_grad(); loss.backward(); optimizer.step()
    scheduler.step()
    print(f"Epoch {epoch} — LR {scheduler.get_last_lr()[0]:.4f}")
```
Why it matters:

- Balances stability (early layers) with adaptability (later layers).

- Often yields better accuracy when you have moderate data.

### Example 3: Swapping in EfficientNet‑B0 & Mixed Precision
Use a different backbone and accelerate training with AMP.

```python
import torch.cuda.amp as amp
from torchvision import models

# 1. Load EfficientNet‑B0
model = models.efficientnet_b0(pretrained=True)

# 2. Freeze backbone, replace classifier
for param in model.features.parameters():
    param.requires_grad = False
model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(train_ds.classes))
model.to(device)

# 3. Optimizer & scaler
optimizer = optim.Adam(model.classifier.parameters(), lr=5e-4)
scaler    = amp.GradScaler()

# 4. Mixed‑precision training loop
for epoch in range(1, 6):
    model.train()
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()

        with amp.autocast():
            logits = model(xb)
            loss   = criterion(logits, yb)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    print(f"Epoch {epoch} (EfficientNet‑B0) complete")
```
Why it matters:

- EfficientNet often outperforms ResNet at similar FLOPs.

- AMP (automatic mixed precision) speeds up training and reduces memory.

## Tips for Transfer Learning Success
- Data Augmentation: Random crops, flips, color jitter—especially important for small datasets.

- Learning Rates: Use a lower LR (1e-4 – 1e-3) when fine‑tuning, and possibly a higher one (1e-3 – 1e-2) for the new head.

Early Stopping: Monitor validation loss/accuracy; stop when you see no improvement.

Batch Normalization: Be cautious when freezing models with BN—either leave them in eval mode or unfreeze their running stats if you have enough data.

Layer-wise LR: You can assign different LRs to different parameter groups (optimizer = SGD([{'params': backbone, 'lr':1e-4}, {'params': head, 'lr':1e-3}])).

Feel free to pick one of these patterns—feature extraction, gradual unfreezing, or backbone swapping—and let me know if you’d like more detail (e.g. full training/validation loops, metric logging, or visualizing learned filters)!
