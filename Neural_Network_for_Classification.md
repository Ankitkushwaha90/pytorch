Here are three progressively richer examples of classification models using PyTorch’s nn.Sequential API. Each illustrates key concepts—from basic feed‑forward networks to convolutional nets—and shows how to load data, define a model, train, and evaluate.

### 1. Binary Classification on Synthetic Data
A simple 2‑feature dataset, label = 1 if x₀ + x₁ > 0, else 0.

```python
import torch
from torch import nn, optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# 1. Prepare data
X = torch.randn(1000, 2)
Y = (X.sum(dim=1) > 0).long()
X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2)

train_ds = TensorDataset(X_train, Y_train)
val_ds   = TensorDataset(X_val,   Y_val)
train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
val_loader   = DataLoader(val_ds,   batch_size=64)

# 2. Define model
model = nn.Sequential(
    nn.Linear(2, 16),
    nn.ReLU(),
    nn.Linear(16, 8),
    nn.ReLU(),
    nn.Linear(8, 2)      # two outputs → logits for classes 0/1
)

# 3. Loss & optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# 4. Training loop
for epoch in range(1, 51):
    model.train()
    for xb, yb in train_loader:
        logits = model(xb)
        loss = criterion(logits, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    # Validation
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for xb, yb in val_loader:
            preds = model(xb).argmax(dim=1)
            all_preds.append(preds)
            all_labels.append(yb)
    acc = accuracy_score(torch.cat(all_labels), torch.cat(all_preds))
    if epoch % 10 == 0:
        print(f"Epoch {epoch:>2} — Val Acc: {acc:.3f}")
```
Key points:

- Uses `nn.Sequential` to stack layers.

- Final layer size = number of classes; no softmax needed before `CrossEntropyLoss`.

- Shows train/validation split and accuracy tracking.

### 2. Multiclass Classification on Iris
Using the classic Iris dataset (3 classes, 4 features).

```python
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# 1. Load & preprocess
iris = pd.read_csv("https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv")
X = torch.tensor(iris.iloc[:, :4].values, dtype=torch.float32)
le = LabelEncoder().fit(iris["species"])
Y = torch.tensor(le.transform(iris["species"]), dtype=torch.long)

# Simple Dataset
class IrisDataset(Dataset):
    def __init__(self, X, Y): self.X, self.Y = X, Y
    def __len__(self):     return len(self.Y)
    def __getitem__(self, i): return self.X[i], self.Y[i]

ds = IrisDataset(X, Y)
loader = DataLoader(ds, batch_size=16, shuffle=True)

# 2. Model
model = nn.Sequential(
    nn.Linear(4, 32),
    nn.ReLU(),
    nn.Linear(32, 16),
    nn.ReLU(),
    nn.Linear(16, 3)
)

# 3. Train
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.05)

for epoch in range(1, 201):
    total_loss = 0
    for xb, yb in loader:
        loss = criterion(model(xb), yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    if epoch % 50 == 0:
        print(f"Epoch {epoch:>3} — Loss: {total_loss/len(loader):.4f}")

# 4. Quick test on entire set
with torch.no_grad():
    preds = model(X).argmax(dim=1)
    acc = (preds == Y).float().mean()
print(f"Iris training accuracy: {acc:.3f}")
```
Key points:

- Shows how to wrap a pandas DataFrame in a custom `Dataset`.

- Multiclass (3 flowers): final layer size = 3.

### 3. CNN for MNIST Digit Classification
A small convolutional network on the MNIST dataset via `torchvision`.

```python
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import accuracy_score

# 1. Data transforms & loaders
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
train_ds = datasets.MNIST(root=".", train=True,  download=True, transform=transform)
test_ds  = datasets.MNIST(root=".", train=False, download=True, transform=transform)
train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
test_loader  = DataLoader(test_ds,  batch_size=1000)

# 2. CNN model via Sequential
model = nn.Sequential(
    # Conv block 1
    nn.Conv2d(1, 16, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(2),        # → 16×14×14

    # Conv block 2
    nn.Conv2d(16, 32, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(2),        # → 32×7×7

    nn.Flatten(),           # → 32*7*7 = 1568
    nn.Linear(1568, 128),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(128, 10)      # 10 classes (digits 0–9)
)

# 3. Loss & optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 4. Training loop
for epoch in range(1, 6):
    model.train()
    for xb, yb in train_loader:
        loss = criterion(model(xb), yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch} complete")

# 5. Evaluation
model.eval()
all_preds, all_labels = [], []
with torch.no_grad():
    for xb, yb in test_loader:
        all_preds.append(model(xb).argmax(dim=1))
        all_labels.append(yb)
acc = accuracy_score(torch.cat(all_labels), torch.cat(all_preds))
print(f"MNIST test accuracy: {acc:.3f}")
```
Key points:

- Uses nn.Conv2d, nn.MaxPool2d, nn.Flatten, nn.Dropout, all inside Sequential.

- Illustrates training on a real vision dataset with normalization, batching, and evaluation metrics.

### Tips for Deepening Your Understanding
1. Visualize Loss & Metrics:

- Plot training vs. validation loss/accuracy curves to detect overfitting or underfitting.

2. Experiment with Hyperparameters:

- Learning rates, batch sizes, network depth, activation functions, optimizers, dropout rates.

3. Extend with Callbacks:

- Add early stopping, learning‑rate schedulers (torch.optim.lr_scheduler).

4. Move to GPU:

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
xb, yb = xb.to(device), yb.to(device)
```
5. Try Transfer Learning:

Use pretrained feature extractors (`torchvision.models`) and fine‑tune on your task.

Feel free to pick one of these and we can dive deeper—e.g., add visualization of decision boundaries, implement training loops with tqdm progress bars, or integrate TensorBoard/Weights & Biases!
