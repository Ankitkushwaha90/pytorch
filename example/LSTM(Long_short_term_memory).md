Below are three progressively richer examples using LSTM layers in PyTorch. Each is designed to highlight different aspects of LSTMs—manual unrolling, handling variable‑length text data, and using LSTMs for sequence regression—so you can build a deep, intuitive understanding.

## 1. Simple LSTMCell Classifier on Synthetic Sequences
Learn what’s happening step‑by‑step by manually unrolling an nn.LSTMCell.

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score

# 1. Synthetic dataset: sequences of 0/1, label = 1 if more 1’s than 0’s
class SeqDataset(Dataset):
    def __init__(self, N=2000, L=30):
        self.X = torch.randint(0, 2, (N, L)).float()  # [0.0 or 1.0]
        self.Y = (self.X.sum(dim=1) > (L/2)).long()
    def __len__(self): return len(self.Y)
    def __getitem__(self, i): return self.X[i], self.Y[i]

train_ds = SeqDataset(2000, 30)
val_ds   = SeqDataset(500, 30)
train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
val_loader   = DataLoader(val_ds,   batch_size=128)

# 2. Model with nn.LSTMCell unrolled in Python
class LSTMCellClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super().__init__()
        self.lstm_cell = nn.LSTMCell(input_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        # x: [batch, seq_len]
        batch_size, seq_len = x.size()
        hx = torch.zeros(batch_size, self.lstm_cell.hidden_size, device=x.device)
        cx = torch.zeros_like(hx)
        for t in range(seq_len):
            # use each timestep as a single feature vector
            hx, cx = self.lstm_cell(x[:, t:t+1], (hx, cx))
        out = self.fc(hx)   # final hidden state → logits
        return out

model = LSTMCellClassifier(input_dim=1, hidden_dim=32, num_classes=2)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# 3. Training loop
for epoch in range(1, 31):
    model.train()
    for xb, yb in train_loader:
        xb, yb = xb.unsqueeze(-1), yb  # xb: [B,L] → [B,L,1]
        logits = model(xb)
        loss = criterion(logits, yb)
        optimizer.zero_grad(); loss.backward(); optimizer.step()
    # validation
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for xb, yb in val_loader:
            xb = xb.unsqueeze(-1)
            preds = model(xb).argmax(dim=1)
            all_preds.append(preds); all_labels.append(yb)
    acc = accuracy_score(torch.cat(all_labels), torch.cat(all_preds))
    if epoch % 10 == 0:
        print(f"Epoch {epoch:>2} — Val Acc: {acc:.3f}")
```
Key takeaways:

- Unrolling with nn.LSTMCell gives you full control over hidden & cell states.

- You see exactly how information flows from one timestep to the next.

- Final hidden state hx is passed through a linear layer for classification.

### 2. LSTM for Sentiment (IMDB) with Packing & Padding
Handle variable‑length text by packing sequences before feeding to nn.LSTM.

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torchtext.datasets import IMDB
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from torch.utils.data import DataLoader

# 1. Build vocab
tokenizer = get_tokenizer("basic_english")
def yield_tokens(data_iter):
    for _, text in data_iter:
        yield tokenizer(text)

train_iter, test_iter = IMDB(split=('train','test'))
vocab = build_vocab_from_iterator(yield_tokens(train_iter), specials=["<pad>","<unk>"])
vocab.set_default_index(vocab["<unk>"])

# 2. Collate fn for variable lengths
def collate(batch):
    texts, labels, lengths = [], [], []
    for label, text in batch:
        ids = torch.tensor(vocab(tokenizer(text)), dtype=torch.long)
        texts.append(ids); lengths.append(len(ids))
        labels.append(1 if label=="pos" else 0)
    texts = pad_sequence(texts, padding_value=vocab["<pad>"])
    return texts, torch.tensor(labels), torch.tensor(lengths)

train_loader = DataLoader(list(train_iter)[:2000], batch_size=32,
                          shuffle=True, collate_fn=collate)
test_loader  = DataLoader(list(test_iter)[:500],  batch_size=64,
                          shuffle=False, collate_fn=collate)

# 3. LSTM model
class LSTMSentiment(nn.Module):
    def __init__(self, vocab_size, emb_dim, hid_dim, num_layers, num_classes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=vocab["<pad>"])
        self.lstm = nn.LSTM(emb_dim, hid_dim, num_layers,
                            batch_first=False, bidirectional=False)
        self.fc = nn.Linear(hid_dim, num_classes)

    def forward(self, text, lengths):
        # text: [seq_len, batch]
        embedded = self.embedding(text)  # [seq_len, B, E]
        packed   = pack_padded_sequence(embedded, lengths.cpu(),
                                        enforce_sorted=False)
        packed_out, (hn, _) = self.lstm(packed)
        # hn: [num_layers, B, hid_dim]; take last layer
        logits = self.fc(hn[-1])
        return logits

model = LSTMSentiment(len(vocab), emb_dim=64, hid_dim=128,
                      num_layers=2, num_classes=2).to(torch.device("cuda"))
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# 4. Training & quick evaluation
for epoch in range(1, 6):
    model.train()
    for text, labels, lengths in train_loader:
        text, labels, lengths = text.to(model.embedding.weight.device), labels.to(model.embedding.weight.device), lengths
        logits = model(text, lengths)
        loss   = criterion(logits, labels)
        optimizer.zero_grad(); loss.backward(); optimizer.step()

    # eval
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for text, labels, lengths in test_loader:
            text, labels = text.to(model.embedding.weight.device), labels.to(model.embedding.weight.device)
            preds = model(text, lengths).argmax(dim=1)
            correct += (preds == labels).sum().item()
            total   += labels.size(0)
    print(f"Epoch {epoch} — Test Acc: {correct/total:.3f}")
```
Key takeaways:

- pad_sequence & pack_padded_sequence let the LSTM ignore padded timesteps.

- You retrieve the final hidden state from the top LSTM layer for classification.

- Handles real‑world variable‑length text smoothly.

### 3. LSTM for Sequence Regression: Sine‑Wave Forecasting
Use an LSTM to predict the next value in a noisy sine wave—demonstrates LSTM for regression.

```python
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

# 1. Generate sine data
T = 1000
data = np.sin(np.linspace(0, 20*np.pi, T)) + 0.1*np.random.randn(T)
seq_len = 30

# 2. Dataset for sliding-window sequences
class SineDataset(Dataset):
    def __init__(self, series, L):
        self.X, self.Y = [], []
        for i in range(len(series) - L):
            self.X.append(series[i:i+L])
            self.Y.append(series[i+L])
        self.X = torch.tensor(self.X, dtype=torch.float32).unsqueeze(-1)  # [N,L,1]
        self.Y = torch.tensor(self.Y, dtype=torch.float32).unsqueeze(-1)  # [N,1]
    def __len__(self): return len(self.Y)
    def __getitem__(self, i): return self.X[i], self.Y[i]

dataset = SineDataset(data, seq_len)
train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

# 3. LSTM regression model
class LSTMForecast(nn.Module):
    def __init__(self, input_dim, hid_dim, num_layers):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hid_dim, num_layers,
                            batch_first=True, bidirectional=False)
        self.fc = nn.Linear(hid_dim, 1)

    def forward(self, x):
        # x: [B, L, 1]
        out, _ = self.lstm(x)        # [B, L, hid_dim]
        last = out[:, -1, :]         # use last timestep
        return self.fc(last)         # [B, 1]

model = LSTMForecast(input_dim=1, hid_dim=50, num_layers=2)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# 4. Train
losses = []
for epoch in range(1, 201):
    model.train()
    epoch_loss = 0
    for xb, yb in train_loader:
        pred = model(xb)
        loss = criterion(pred, yb)
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        epoch_loss += loss.item()
    losses.append(epoch_loss / len(train_loader))
    if epoch % 50 == 0:
        print(f"Epoch {epoch} — MSE: {losses[-1]:.4f}")

# 5. Plot training loss
plt.plot(losses)
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.title("LSTM Sine‑Wave Forecasting Loss")
plt.show()
```
Key takeaways:

- Shows LSTM for regression, not just classification.

- Uses only the last hidden state (out[:, -1, :]) for prediction.

- Visualizing the loss curve helps you see convergence dynamics.

## Next‑Step Suggestions
- Visualize hidden/cell states over time to see how the LSTM “remembers.”

- Compare LSTM vs. GRU on the same task to see performance and speed trade‑offs.

- Stack more layers or add dropout (nn.Dropout) between layers.

- Use bidirectional LSTM for tasks where future context helps (e.g., tagging).

- Hyperparameter sweep: vary hidden size, number of layers, learning rate, etc., and plot results.

Pick any of these examples to tinker with—and let me know if you want deeper dives (e.g., plotting state activations, integrating attention on top of LSTM, or applying these to real datasets)!
