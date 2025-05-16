Here are multiple examples of custom loss functions in PyTorch, ranging from simple to advanced use cases. These examples will help you understand how to define and use custom loss functions in real projects.

### ✅ 1. Basic Custom MSE (Manual Calculation)
Implementing Mean Squared Error manually, without using nn.MSELoss.

```python
import torch
import torch.nn as nn

def custom_mse_loss(pred, target):
    return torch.mean((pred - target) ** 2)

# Example
y_pred = torch.tensor([2.5, 0.0, 2.0], requires_grad=True)
y_true = torch.tensor([3.0, -0.5, 2.0])
loss = custom_mse_loss(y_pred, y_true)
print("Custom MSE Loss:", loss.item())
```
### ✅ 2. Custom Class-Based Loss Function (MAE)
Use an object-oriented way (extending nn.Module) to define loss.

```python
class CustomMAELoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, target):
        return torch.mean(torch.abs(pred - target))

# Example
loss_fn = CustomMAELoss()
pred = torch.tensor([1.5, 2.0], requires_grad=True)
true = torch.tensor([1.0, 2.5])
loss = loss_fn(pred, true)
print("MAE Loss:", loss.item())
```
### ✅ 3. Huber Loss (Smooth L1 Loss)
Good for outlier-sensitive regression.

```python
class HuberLoss(nn.Module):
    def __init__(self, delta=1.0):
        super().__init__()
        self.delta = delta

    def forward(self, pred, target):
        error = pred - target
        cond = torch.abs(error) < self.delta
        squared_loss = 0.5 * error ** 2
        linear_loss = self.delta * (torch.abs(error) - 0.5 * self.delta)
        return torch.mean(torch.where(cond, squared_loss, linear_loss))

# Example
loss_fn = HuberLoss(delta=1.0)
pred = torch.tensor([2.5, 0.0], requires_grad=True)
target = torch.tensor([3.0, -1.0])
loss = loss_fn(pred, target)
print("Huber Loss:", loss.item())
```
### ✅ 4. Focal Loss (for imbalanced classification)
Useful in cases where class imbalance hurts standard CrossEntropy.

```python
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.bce = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, logits, targets):
        bce_loss = self.bce(logits, targets)
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        return torch.mean(focal_loss)

# Example for binary classification
logits = torch.tensor([0.2, 1.5], requires_grad=True)
targets = torch.tensor([0., 1.])
loss_fn = FocalLoss(alpha=0.25, gamma=2)
loss = loss_fn(logits, targets)
print("Focal Loss:", loss.item())
```
### ✅ 5. Contrastive Loss
Used in Siamese Networks for learning similarity.

```python
class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = torch.nn.functional.pairwise_distance(output1, output2)
        loss = (1 - label) * euclidean_distance ** 2 + label * torch.clamp(self.margin - euclidean_distance, min=0.0) ** 2
        return torch.mean(loss)

# Example (toy outputs and labels)
output1 = torch.tensor([[1.0, 2.0]], requires_grad=True)
output2 = torch.tensor([[1.1, 2.1]], requires_grad=True)
label = torch.tensor([0.0])  # 0 = similar, 1 = dissimilar
loss_fn = ContrastiveLoss()
loss = loss_fn(output1, output2, label)
print("Contrastive Loss:", loss.item())
```
### ✅ 6. Custom Loss with Regularization (L2 penalty on weights)
```python
def l2_regularization(model, lambda_reg=1e-4):
    l2_norm = sum(param.pow(2.0).sum() for param in model.parameters())
    return lambda_reg * l2_norm

def custom_loss_with_l2(pred, target, model, lambda_reg=1e-4):
    mse = torch.mean((pred - target) ** 2)
    l2 = l2_regularization(model, lambda_reg)
    return mse + l2

# Example
class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(2, 1)

    def forward(self, x):
        return self.fc(x)

model = DummyModel()
x = torch.tensor([[1.0, 2.0]])
y_true = torch.tensor([[3.0]])
y_pred = model(x)
loss = custom_loss_with_l2(y_pred, y_true, model, lambda_reg=0.01)
print("Custom MSE + L2 Loss:", loss.item())
```
### 🧠 Bonus: Combine Multiple Losses
Sometimes, models optimize multiple objectives (multi-task).

```python
def combined_loss(pred_class, true_class, pred_reg, true_reg):
    ce = nn.CrossEntropyLoss()(pred_class, true_class)
    mse = nn.MSELoss()(pred_reg, true_reg)
    return ce + 0.5 * mse  # weighted sum

# Use this when your model outputs both class and regression values.
```
## 📘 Summary Table

| Name            | Type         | Use Case                                        |
|-----------------|--------------|--------------------------------------------------|
| Custom MSE      | Regression   | Basic understanding                              |
| Custom MAE      | Regression   | Robust to outliers                               |
| Huber Loss      | Regression   | Combination of MSE + MAE                         |
| Focal Loss      | Classification | Handle class imbalance                        |
| Contrastive Loss| Similarity   | Siamese networks                                 |
| L2 Regularized  | Any          | Add penalty to model parameters                  |
| Combined Loss   | Multi-task   | Multi-output models (e.g. class + reg)           |

---

If you want more domain-specific custom losses (e.g. **NLP**, **vision**, **adversarial training**), I can provide examples tailored to that too.
