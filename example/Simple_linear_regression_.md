Great! You're on the right path learning PyTorch through practical examples like simple linear regression. Here's a list of beginner-friendly yet insightful examples that build understanding of PyTorch fundamentals like tensors, autograd, nn.Module, optim, etc.

### ✅ 1. Simple Linear Regression (Already Done by You)
Predict `Y = 3.5X + 2.0` using one linear neuron.

### ✅ 2. Polynomial Regression (Non-linear data with Linear Model)
```python
import torch
import torch.nn as nn
import torch.optim as optim

# Generate non-linear data
X = torch.linspace(-3, 3, 100).unsqueeze(1)
Y = 0.5 * X**2 + X + 2 + 0.1 * torch.randn(X.size())

# Define a polynomial model using linear layer and input powers
class PolynomialRegression(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(3, 1)  # x, x^2, x^3

    def forward(self, x):
        x2 = x ** 2
        x3 = x ** 3
        features = torch.cat([x, x2, x3], dim=1)
        return self.linear(features)

model = PolynomialRegression()
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Training
for epoch in range(1000):
    model.train()
    pred = model(X)
    loss = criterion(pred, Y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 100 == 0:
        print(f'Epoch {epoch}, Loss: {loss.item()}')
```
### ✅ 3. Logistic Regression (Binary Classification)
```python
import torch
import torch.nn as nn
import torch.optim as optim

# Binary data
X = torch.randn(100, 2)
Y = (X[:, 0] + X[:, 1] > 0).float().unsqueeze(1)  # Label: 1 if sum > 0 else 0

model = nn.Sequential(nn.Linear(2, 1), nn.Sigmoid())
criterion = nn.BCELoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

for epoch in range(1000):
    pred = model(X)
    loss = criterion(pred, Y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item()}")
```
### ✅ 4. Multi-class Classification (Softmax Output)
```python
import torch
import torch.nn as nn
import torch.optim as optim

X = torch.randn(100, 4)
Y = torch.randint(0, 3, (100,))  # 3 classes: 0, 1, 2

model = nn.Linear(4, 3)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

for epoch in range(300):
    pred = model(X)
    loss = criterion(pred, Y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if epoch % 50 == 0:
        print(f'Epoch {epoch}, Loss: {loss.item()}')
```
### ✅ 5. Custom Dataset with DataLoader
```python
from torch.utils.data import Dataset, DataLoader

class CustomDataset(Dataset):
    def __init__(self):
        self.X = torch.randn(100, 1)
        self.Y = 3.5 * self.X + 2.0

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

dataset = CustomDataset()
loader = DataLoader(dataset, batch_size=10, shuffle=True)

model = nn.Linear(1, 1)
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

for epoch in range(300):
    for batch_X, batch_Y in loader:
        pred = model(batch_X)
        loss = criterion(pred, batch_Y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if epoch % 50 == 0:
        print(f'Epoch {epoch}, Loss: {loss.item()}')
```
### ✅ 6. Manual Backprop (No nn.Module)
```python
X = torch.randn(100, 1)
Y = 2.5 * X + 1.5

w = torch.randn(1, requires_grad=True)
b = torch.randn(1, requires_grad=True)
lr = 0.01

for epoch in range(300):
    pred = w * X + b
    loss = ((pred - Y) ** 2).mean()
    
    loss.backward()
    
    with torch.no_grad():
        w -= lr * w.grad
        b -= lr * b.grad
        w.grad.zero_()
        b.grad.zero_()
    
    if epoch % 50 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item()}")
```
### ✅ 7. XOR Neural Network (Non-linearly separable)
```python
X = torch.tensor([[0,0],[0,1],[1,0],[1,1]]).float()
Y = torch.tensor([[0],[1],[1],[0]]).float()

model = nn.Sequential(
    nn.Linear(2, 4),
    nn.ReLU(),
    nn.Linear(4, 1),
    nn.Sigmoid()
)

criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.1)

for epoch in range(5000):
    pred = model(X)
    loss = criterion(pred, Y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if epoch % 500 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item()}")
```
### 💡 Bonus Topics You Can Try After:
- Linear regression with PyTorch Lightning

- Image classification using torchvision datasets

- CNN for MNIST classification

- RNN or LSTM for sequence modeling
