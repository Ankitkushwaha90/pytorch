##### Here are multiple examples of saving and loading models in PyTorch, covering:

- Saving entire model vs state_dict

- Loading for inference and resuming training

- Saving checkpoints

- Saving custom models

These examples aim to give you a clear and complete understanding of best practices for saving/loading models.

### ✅ 1. Save and Load Entire Model (Not Recommended for Deployment)
```python
import torch
import torch.nn as nn

# Define model
model = nn.Linear(2, 1)

# Save entire model
torch.save(model, "full_model.pth")

# Load entire model
loaded_model = torch.load("full_model.pth")
print(loaded_model)
```
| 🔴 Not recommended for production due to dependency on class definition structure.

## ✅ 2. Save and Load `state_dict` (Recommended Practice)
```python
# Define model
model = nn.Linear(2, 1)

# Save model state_dict
torch.save(model.state_dict(), "model_weights.pth")

# To load
model2 = nn.Linear(2, 1)
model2.load_state_dict(torch.load("model_weights.pth"))
model2.eval()
```
| ✅ Best practice for deployment. Always use state_dict() for saving/loading.

✅ 3. Save and Load with Optimizer (For Resuming Training)
```python
model = nn.Linear(2, 1)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# Save both
torch.save({
    'epoch': 10,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': 0.02,
}, 'checkpoint.pth')

# Load both
checkpoint = torch.load('checkpoint.pth')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
start_epoch = checkpoint['epoch']
loss = checkpoint['loss']

model.train()  # Resume training
```
### ✅ 4. Save & Load Custom Model Class
```python
class CustomModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(2, 1)

    def forward(self, x):
        return self.fc(x)

model = CustomModel()
torch.save(model.state_dict(), "custom_model.pth")

# To load
loaded = CustomModel()
loaded.load_state_dict(torch.load("custom_model.pth"))
loaded.eval()
```
### ✅ 5. Save and Load on CPU/GPU Selectively
```python
# Save on GPU
torch.save(model.state_dict(), 'model_gpu.pth')

# Load on CPU
device = torch.device('cpu')
model.load_state_dict(torch.load('model_gpu.pth', map_location=device))
```
| ✅ Useful when training on GPU and deploying on CPU.

### ✅ 6. Save Only Best Model (Based on Validation Loss)
```python
best_loss = float('inf')

for epoch in range(10):
    # Train...
    val_loss = 0.02 * epoch  # dummy
    
    if val_loss < best_loss:
        best_loss = val_loss
        torch.save(model.state_dict(), "best_model.pth")
        print(f"Saved best model at epoch {epoch}")
```
### ✅ 7. Save and Load Multiple Models (e.g. GAN, multi-task)
```python
generator = nn.Linear(10, 1)
discriminator = nn.Linear(1, 1)

torch.save({
    'generator': generator.state_dict(),
    'discriminator': discriminator.state_dict()
}, 'gan_models.pth')

# Load
gen2 = nn.Linear(10, 1)
disc2 = nn.Linear(1, 1)

checkpoint = torch.load('gan_models.pth')
gen2.load_state_dict(checkpoint['generator'])
disc2.load_state_dict(checkpoint['discriminator'])
```
### ✅ 8. Save Training Checkpoint Every Epoch
```python
for epoch in range(5):
    # Train...

    torch.save({
        'epoch': epoch,
        'model': model.state_dict(),
    }, f"checkpoint_epoch_{epoch}.pth")
```
| 🔄 Helps in debugging or resuming from mid-point.

### ✅ 9. Export to ONNX (For Interoperability)
```python
dummy_input = torch.randn(1, 2)
torch.onnx.export(model, dummy_input, "model.onnx", input_names=['input'], output_names=['output'])
```
| 🧠 Use when integrating PyTorch with C++, Java, or mobile.

### ✅ 10. Save Intermediate Layer Outputs for Later Use
```python
features = []

def hook_fn(module, input, output):
    features.append(output)

hook = model.register_forward_hook(hook_fn)

# Forward pass
input = torch.randn(1, 2)
output = model(input)

# Save feature
torch.save(features[0], "intermediate_output.pth")
```
## 🧠 Summary of Best Practices

| Task                       | Best Method                                   |
|----------------------------|-----------------------------------------------|
| Saving model for inference | `torch.save(model.state_dict())`              |
| Resuming training          | Save both model & optimizer `state_dict`      |
| Deploying on CPU from GPU | Use `map_location=torch.device('cpu')`        |
| Multiple models            | Use dictionary with separate keys             |
| Production deployment      | Prefer **ONNX** or **TorchScript** for safety |

---

Would you like me to show saving/loading in a full classification training loop too?
