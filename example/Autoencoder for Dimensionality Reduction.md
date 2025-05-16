Below are three progressively richer autoencoder examples for dimensionality reduction. Each one deepens your understanding of how autoencoders learn compact representations:

## 1. Simple Linear Autoencoder on Synthetic Data
Compress 2‑D points into a 1‑D “bottleneck” and reconstruct.

```python
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# 1. Generate synthetic 2D data (points on a noisy line)
X = torch.randn(1000, 2) * 0.5 + torch.tensor([2.0, -1.0])
X += 0.1 * torch.randn_like(X)

# 2. Define a linear autoencoder: 2 → 1 → 2
class LinearAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Linear(2, 1)
        self.decoder = nn.Linear(1, 2)

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat, z

model = LinearAE()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)


# 3. Train
for epoch in range(200):
    optimizer.zero_grad()
    X_hat, Z = model(X)
    loss = criterion(X_hat, X)
    loss.backward()
    optimizer.step()
    if epoch % 50 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

# 4. Visualize original vs reconstructed and bottleneck
with torch.no_grad():
    X_hat, Z = model(X)
plt.figure(figsize=(12,4))
plt.subplot(1,3,1)
plt.scatter(X[:,0], X[:,1], s=5); plt.title("Original")
plt.subplot(1,3,2)
plt.scatter(X_hat[:,0], X_hat[:,1], s=5); plt.title("Reconstructed")
plt.subplot(1,3,3)
plt.scatter(Z, torch.zeros_like(Z), s=5); plt.title("1D Bottleneck")
plt.tight_layout()
plt.show()
```
What to learn:

- A purely linear AE here is equivalent to PCA—the 1‑D encoder finds the principal component.

- Visualizing Z shows how the network arranges points along a single axis.

## 2. Deep MLP Autoencoder on MNIST
Compress 28×28 images into a 32‑dim latent vector and reconstruct.

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# 1. Data loading
transform = transforms.ToTensor()
train_ds = datasets.MNIST(".", train=True, download=True, transform=transform)
loader   = DataLoader(train_ds, batch_size=128, shuffle=True)

# 2. Define MLP Autoencoder: 784 → 128 → 32 → 128 → 784
class MNIST_AE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(32, 128),
            nn.ReLU(),
            nn.Linear(128, 784),
            nn.Sigmoid(),        # pixels in [0,1]
            nn.Unflatten(1, (1,28,28))
        )

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat, z

model = MNIST_AE().to("cuda")
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# 3. Training
for epoch in range(1, 11):
    model.train()
    total_loss = 0
    for xb, _ in loader:
        xb = xb.to("cuda")
        xb_hat, _ = model(xb)
        loss = criterion(xb_hat, xb)
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch}, Loss: {total_loss/len(loader):.4f}")

# 4. Show reconstructions
model.eval()
with torch.no_grad():
    xb, _ = next(iter(loader))
    xb, xb_hat, z = xb.to("cuda"), *model(xb.to("cuda"))
    xb, xb_hat = xb.cpu(), xb_hat.cpu()

fig, axes = plt.subplots(2, 10, figsize=(12,3))
for i in range(10):
    axes[0,i].imshow(xb[i,0], cmap='gray'); axes[0,i].axis('off')
    axes[1,i].imshow(xb_hat[i,0], cmap='gray'); axes[1,i].axis('off')
axes[0,0].set_title("Original"); axes[1,0].set_title("Reconstructed")
plt.show()
```
What to learn:

- Nonlinear encoder/decoder can capture far richer structure than PCA.

- Inspecting the 32‑dim z (e.g., t‑SNE or scatterplots) reveals class clusters.

## 3. Convolutional Autoencoder on CIFAR‑10
Use conv/deconv layers to compress 32×32×3 images into a spatial bottleneck.

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# 1. Data loaders with normalization
tf = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
])
train_ds = datasets.CIFAR10(".", train=True, download=True, transform=tf)
loader   = DataLoader(train_ds, batch_size=128, shuffle=True)

# 2. Conv Autoencoder: enc 3×32×32 → 16×8×8 → z (16×8×8) → dec → 3×32×32
class ConvAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 8, 3, stride=2, padding=1),  # →8×16×16
            nn.ReLU(),
            nn.Conv2d(8,16,3, stride=2, padding=1),   # →16×8×8
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(16, 8, 3, stride=2, padding=1, output_padding=1),  #→8×16×16
            nn.ReLU(),
            nn.ConvTranspose2d(8, 3, 3, stride=2, padding=1, output_padding=1),   #→3×32×32
            nn.Tanh()  # outputs in [-1,1]
        )

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat, z

model = ConvAE().to("cuda")
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# 3. Training
for epoch in range(1, 16):
    model.train()
    total_loss = 0
    for xb, _ in loader:
        xb = xb.to("cuda")
        xb_hat, _ = model(xb)
        loss = criterion(xb_hat, xb)
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch}, Loss: {total_loss/len(loader):.4f}")

# 4. Visualize reconstructions
model.eval()
with torch.no_grad():
    xb, _ = next(iter(loader))
    xb, xb_hat = xb.to("cuda"), model(xb.to("cuda"))[0]
    xb, xb_hat = xb[:8].cpu(), xb_hat[:8].cpu()

fig, axes = plt.subplots(2, 8, figsize=(12,4))
for i in range(8):
    # denormalize for display
    orig = xb[i].permute(1,2,0)*0.5 + 0.5
    recon = xb_hat[i].permute(1,2,0)*0.5 + 0.5
    axes[0,i].imshow(orig); axes[0,i].axis('off')
    axes[1,i].imshow(recon); axes[1,i].axis('off')
axes[0,0].set_title("Original"); axes[1,0].set_title("Reconstructed")
plt.show()
```
## 🚀 Bonus: Variational Autoencoder (VAE)
If you’re curious about generative dimensionality reduction, explore a VAE where the encoder outputs a mean & log‑variance, and you sample a latent z. This introduces the KL‑divergence term into the loss and yields a smoothly interpolatable latent space.

## Further Exploration
- Latent-space visualization: Apply t‑SNE or PCA to your bottleneck features z to see class/group structure.

- Denoising autoencoder: Train on noisy inputs but reconstruct the clean data.

Sparse autoencoder: Add an L1 penalty on z to encourage sparsity.

Contractive autoencoder: Penalize the Jacobian of the encoder for robustness.

Pick an example and tinker—plot loss curves, vary the bottleneck size, or switch activation functions—to internalize how autoencoders learn compact, meaningful representations!
