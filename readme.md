
Here is a collection of PyTorch code examples, ranging from basic concepts to advanced neural network architectures. These examples are intended to guide you through the process of learning PyTorch.

### 1. Basic Tensor Operations
```python
import torch

# Create tensors
a = torch.tensor([[1, 2], [3, 4]])
b = torch.tensor([[5, 6], [7, 8]])

# Element-wise addition
add_result = a + b
print("Addition:", add_result)

# Element-wise multiplication
mul_result = a * b
print("Multiplication:", mul_result)

# Matrix multiplication
matmul_result = torch.matmul(a, b)
print("Matrix Multiplication:", matmul_result)
```
### 2. Simple Linear Regression in PyTorch
```python
import torch
import torch.nn as nn
import torch.optim as optim

# Generate random data
X = torch.randn(100, 1)
Y = 3.5 * X + 2.0

# Define a simple linear regression model
model = nn.Linear(1, 1)

# Loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Training loop
for epoch in range(1000):
    model.train()
    
    # Forward pass
    pred = model(X)
    loss = criterion(pred, Y)
    
    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if epoch % 100 == 0:
        print(f'Epoch {epoch}, Loss: {loss.item()}')
```
### 3. Neural Network for Classification (Using Sequential API)
```python
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

# Load dataset (MNIST)
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)

# Define a simple feedforward neural network
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28*28, 128)
        self.fc2 = nn.Linear(128, 10)
    
    def forward(self, x):
        x = self.flatten(x)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = SimpleNN()

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
for epoch in range(5):
    for data, target in train_loader:
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

    print(f'Epoch {epoch + 1}, Loss: {loss.item()}')
```
### 4. Convolutional Neural Network (CNN) for Image Classification
```python
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

# Load dataset (CIFAR-10)
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)

# Define CNN model
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.fc1 = nn.Linear(64 * 6 * 6, 512)
        self.fc2 = nn.Linear(512, 10)
    
    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 64 * 6 * 6)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = CNN()

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
for epoch in range(10):
    for data, target in train_loader:
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

    print(f'Epoch {epoch + 1}, Loss: {loss.item()}')
```
### 5. Recurrent Neural Network (RNN) for Sequence Classification
```python
import torch
import torch.nn as nn
import torch.optim as optim
from torchtext.datasets import IMDB
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

# Tokenizer and vocabulary
tokenizer = get_tokenizer("basic_english")
train_iter, _ = IMDB()

# Build vocabulary from training data
def yield_tokens(data_iter):
    for _, text in data_iter:
        yield tokenizer(text)

vocab = build_vocab_from_iterator(yield_tokens(train_iter), specials=["<unk>"])
vocab.set_default_index(vocab["<unk>"])

# Simple RNN Model
class RNN(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, output_size):
        super(RNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.RNN(embed_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.embedding(x)
        x, _ = self.rnn(x)
        x = self.fc(x[:, -1, :])
        return x

# Hyperparameters
vocab_size = len(vocab)
embed_size = 128
hidden_size = 128
output_size = 2

model = RNN(vocab_size, embed_size, hidden_size, output_size)

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Dummy data (preprocessed)
# Replace this with real text data preprocessing using tokenization and padding
X = torch.randint(0, vocab_size, (32, 100))  # Randomly generated input (batch_size, seq_len)
Y = torch.randint(0, 2, (32,))  # Randomly generated target (batch_size)

# Training loop
for epoch in range(5):
    optimizer.zero_grad()
    output = model(X)
    loss = criterion(output, Y)
    loss.backward()
    optimizer.step()
    print(f'Epoch {epoch + 1}, Loss: {loss.item()}')
```
### 6. LSTM (Long Short-Term Memory) Network
```python
import torch
import torch.nn as nn
import torch.optim as optim

# Simple LSTM model
class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, output_size):
        super(LSTMModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.embedding(x)
        x, _ = self.lstm(x)
        x = self.fc(x[:, -1, :])
        return x

# Hyperparameters
vocab_size = 5000
embed_size = 128
hidden_size = 128
output_size = 2

model = LSTMModel(vocab_size, embed_size, hidden_size, output_size)

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Dummy data
X = torch.randint(0, vocab_size, (32, 100))  # Randomly generated input
Y = torch.randint(0, 2, (32,))  # Randomly generated target

# Training loop
for epoch in range(5):
    optimizer.zero_grad()
    output = model(X)
    loss = criterion(output, Y)
    loss.backward()
    optimizer.step()
    print(f'Epoch {epoch + 1}, Loss: {loss.item()}')
```
### 7. Transfer Learning with Pre-trained Models (ResNet)
```python
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, datasets, transforms

# Load a pre-trained ResNet model
resnet = models.resnet18(pretrained=True)

# Freeze the base layers
for param in resnet.parameters():
    param.requires_grad = False

# Modify the fully connected layer for new task
num_ftrs = resnet.fc.in_features
resnet.fc = nn.Linear(num_ftrs, 10)  # Assume 10 output classes

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(resnet.fc.parameters(), lr=0.001)

# Example training loop (load real dataset instead of dummy data)
X = torch.randn(32, 3, 224, 224)  # Dummy image data (batch_size, channels, height, width)
Y = torch.randint(0, 10, (32,))   # Dummy target data

for epoch in range(5):
    optimizer.zero_grad()
    output = resnet(X)
    loss = criterion(output, Y)
    loss.backward()
    optimizer.step()
    print(f'Epoch {epoch + 1}, Loss: {loss.item()}')
```
### 8. Autoencoder for Dimensionality Reduction
```python
import torch
import torch.nn as nn
import torch.optim as optim

# Define Autoencoder
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Linear(256, 64)
        )
        self.decoder = nn.Sequential(
            nn.Linear(64, 256),
            nn.ReLU(),
            nn.Linear(256, 784),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# Instantiate model, loss function, and optimizer
model = Autoencoder()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Dummy data (Use actual dataset like MNIST)
X = torch.randn(32, 784)  # Randomly generated input (batch_size, input_dim)

# Training loop
for epoch in range(50):
    optimizer.zero_grad()
    output = model(X)
    loss = criterion(output, X)
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print(f'Epoch {epoch}, Loss: {loss.item()}')
```
### 9. Custom Loss Function
```python
import torch
import torch.nn as nn
import torch.optim as optim

# Custom loss function: Mean Squared Error
def custom_loss(output, target):
    return torch.mean((output - target) ** 2)

# Define a simple model
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(10, 1)

    def forward(self, x):
        return self.fc(x)

model = SimpleModel()

# Optimizer
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Dummy data
X = torch.randn(32, 10)  # Randomly generated input
Y = torch.randn(32, 1)   # Randomly generated target

# Training loop
for epoch in range(5):
    optimizer.zero_grad()
    output = model(X)
    loss = custom_loss(output, Y)
    loss.backward()
    optimizer.step()
    print(f'Epoch {epoch + 1}, Loss: {loss.item()}')
```
### 10. Saving and Loading Models
```python
import torch
import torch.nn as nn

# Define a simple model
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(10, 1)

    def forward(self, x):
        return self.fc(x)

model = SimpleModel()

# Save the model
torch.save(model.state_dict(), 'simple_model.pth')

# Load the model
loaded_model = SimpleModel()
loaded_model.load_state_dict(torch.load('simple_model.pth'))
loaded_model.eval()

# Test loaded model with dummy data
X = torch.randn(1, 10)
output = loaded_model(X)
print(output)
```
These PyTorch examples cover basic operations, simple neural networks, convolutional neural networks (CNNs), recurrent neural networks (RNNs), LSTMs, transfer learning, autoencoders, and saving/loading models. They are designed to help you understand PyTorch's API and structure.
