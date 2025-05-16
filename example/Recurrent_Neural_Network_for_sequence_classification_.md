Below are three progressively richer examples of RNN‑based sequence classification in PyTorch. Each builds on the last to deepen your intuition about recurrent layers, handling variable lengths, and using more powerful cells (LSTM, GRU).

## 1. Simple RNN on Synthetic Binary Sequences
Classify sequences of 0/1 tokens by whether they contain more 1’s than 0’s.

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score

# 1. Synthetic dataset
class SyntheticSeqDataset(Dataset):
    def __init__(self, num_samples=1000, seq_len=20):
        self.X = torch.randint(0, 2, (num_samples, seq_len))    # 0 or 1 tokens
        # label = 1 if sum of tokens > seq_len/2 else 0
        self.Y = (self.X.sum(dim=1) > (seq_len // 2)).long()
    def __len__(self): return len(self.Y)
    def __getitem__(self, i): return self.X[i], self.Y[i]

train_ds = SyntheticSeqDataset(2000, seq_len=20)
val_ds   = SyntheticSeqDataset(500,  seq_len=20)
train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
val_loader   = DataLoader(val_ds,   batch_size=128)

# 2. Model using nn.RNNCell inside a loop (manual unroll)
class SimpleRNNClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super().__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(2, input_size)       # embed token 0/1
        self.rnn_cell  = nn.RNNCell(input_size, hidden_size)
        self.fc        = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # x: [batch, seq_len]
        h = torch.zeros(x.size(0), self.hidden_size, device=x.device)
        for t in range(x.size(1)):
            xt = self.embedding(x[:, t])
            h = self.rnn_cell(xt, h)
        out = self.fc(h)
        return out

model = SimpleRNNClassifier(input_size=8, hidden_size=16, num_classes=2)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# 3. Training loop
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
for epoch in range(1, 31):
    model.train()
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        loss = criterion(logits, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Validation
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for xb, yb in val_loader:
            xb = xb.to(device)
            out = model(xb).argmax(dim=1).cpu()
            preds.append(out); labels.append(yb)
    acc = accuracy_score(torch.cat(labels), torch.cat(preds))
    if epoch % 10 == 0:
        print(f"Epoch {epoch:>2} – Val Acc: {acc:.3f}")
```
Key points:

- Manual unrolling with nn.RNNCell to see step‑by‑step recurrence.

- Uses an Embedding for discrete tokens.

- Final hidden state drives classification.

## 2. LSTM for Sentiment Classification on IMDb Subset
Use nn.LSTM to classify movie reviews (positive/negative) with padding and packing.

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torchtext.datasets import IMDB
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from torch.utils.data import DataLoader

# 1. Prepare data (small subset)
tokenizer = get_tokenizer("basic_english")
def yield_tokens(data_iter):
    for label, text in data_iter:
        yield tokenizer(text)

train_iter, test_iter = IMDB(split=('train', 'test'))
vocab = build_vocab_from_iterator(yield_tokens(train_iter), specials=["<pad>","<unk>"])
vocab.set_default_index(vocab["<unk>"])

def collate_batch(batch):
    texts, labels, lengths = [], [], []
    for label, text in batch:
        tokens = torch.tensor(vocab(tokenizer(text)), dtype=torch.long)
        texts.append(tokens); lengths.append(len(tokens))
        labels.append(1 if label=="pos" else 0)
    texts = pad_sequence(texts, padding_value=vocab["<pad>"])
    return texts, torch.tensor(labels), torch.tensor(lengths)

train_loader = DataLoader(list(train_iter)[:2000], batch_size=32, shuffle=True, collate_fn=collate_batch)
test_loader  = DataLoader(list(test_iter)[:500],  batch_size=64, collate_fn=collate_batch)

# 2. LSTM model
class LSTMSentiment(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers, num_classes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=vocab["<pad>"])
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers, batch_first=False)
        self.fc   = nn.Linear(hidden_dim, num_classes)

    def forward(self, text, lengths):
        # text: [seq_len, batch]
        embedded = self.embedding(text)           # [seq_len, batch, embed_dim]
        packed   = pack_padded_sequence(embedded, lengths.cpu(), enforce_sorted=False)
        packed_out, (hn, _) = self.lstm(packed)
        # hn: [num_layers, batch, hidden_dim]; take last layer
        out = self.fc(hn[-1])
        return out

model = LSTMSentiment(len(vocab), embed_dim=64, hidden_dim=128, num_layers=2, num_classes=2)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 3. Training & evaluation
for epoch in range(1, 6):
    model.train()
    for text, labels, lengths in train_loader:
        text, labels, lengths = text.to(device), labels.to(device), lengths.to(device)
        logits = model(text, lengths)
        loss = criterion(logits, labels)
        optimizer.zero_grad(); loss.backward(); optimizer.step()

    # quick test
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for text, labels, lengths in test_loader:
            text, labels, lengths = text.to(device), labels.to(device), lengths.to(device)
            preds = model(text, lengths).argmax(dim=1)
            correct += (preds == labels).sum().item()
            total   += labels.size(0)
    print(f"Epoch {epoch} – Test Acc: {correct/total:.3f}")
```
Key points:

Uses pad_sequence + pack_padded_sequence to handle variable lengths.

nn.LSTM returns final hidden state hn, which feeds the classifier.

## 3. GRU with Attention for DNA Sequence Classification
Classify synthetic DNA (A/C/G/T) strings into motif‑present vs. absent, with a simple attention layer.

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# 1. Synthetic DNA dataset
class DnaDataset(Dataset):
    def __init__(self, num_samples=1000, seq_len=50):
        self.mapping = {'A':0,'C':1,'G':2,'T':3}
        self.X, self.Y = [], []
        for _ in range(num_samples):
            seq = [self.mapping[ch] for ch in torch.choice(list(self.mapping.keys()), (seq_len,))]
            label = int("ACGT" in "".join(self.mapping.keys()))  # dummy motif logic
            self.X.append(torch.tensor(seq))
            self.Y.append(label)
    def __len__(self): return len(self.Y)
    def __getitem__(self,i): return self.X[i], self.Y[i]

ds = DnaDataset(2000, seq_len=50)
loader = DataLoader(ds, batch_size=32, shuffle=True)

# 2. GRU + Attention model
class GRUAttentionClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes):
        super().__init__()
        self.embed     = nn.Embedding(vocab_size, embed_dim)
        self.gru       = nn.GRU(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.attn_fc   = nn.Linear(hidden_dim*2, 1)
        self.classifier= nn.Linear(hidden_dim*2, num_classes)

    def forward(self, x):
        # x: [batch, seq_len]
        e = self.embed(x)                            # [batch, seq, embed_dim]
        out, _ = self.gru(e)                         # [batch, seq, 2*hidden_dim]
        # Attention weights
        attn_scores = self.attn_fc(out).squeeze(-1)  # [batch, seq]
        weights     = torch.softmax(attn_scores, dim=1).unsqueeze(-1)
        # Context vector
        context = torch.sum(out * weights, dim=1)    # [batch, 2*hidden_dim]
        return self.classifier(context)

model = GRUAttentionClassifier(vocab_size=4, embed_dim=16, hidden_dim=32, num_classes=2)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 3. Train
for epoch in range(1, 11):
    model.train()
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        loss = criterion(logits, yb)
        optimizer.zero_grad(); loss.backward(); optimizer.step()
    if epoch % 5 == 0:
        print(f"Epoch {epoch} complete")
```
Key points:

- Bidirectional GRU captures context from both directions.

- Simple attention mechanism: learns to weight timesteps before pooling.

- Final context vector drives classification.

## Next Steps & Exploration
- Visualize learned attention weights over time to see which positions the model focuses on.

- Compare RNN vs. LSTM vs. GRU on your tasks.

- Stack multiple recurrent layers or add residual connections.

- Incorporate pretrained embeddings (e.g., GloVe) for NLP tasks.

- Benchmark speed vs. accuracy, and try CuDNN‑optimized layers by using nn.LSTM/nn.GRU directly.

Let me know which example you'd like to dive deeper into—e.g., plotting attention maps, benchmarking on real data, or extending to sequence labeling!
