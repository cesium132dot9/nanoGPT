import torch
import torch.nn as nn
from torch.nn import functional as F

# hyperparameter
batch_size = 32
block_size = 8
max_iters = 3000
eval_interval = 300
learning_rate = 1e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 32
# --------------

torch.manual_seed(1337)

# wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
with open('input.txt', 'r', encoding='utf-8') as f: 
    text = f.read()

# all the unique characters that appear in text
chars = sorted(list(set(text)))
vocab_size = len(chars)
# creating a mapping 
stoi = {char : i for i, char in enumerate(chars)}
itos = {i : char for i, char in enumerate(chars)}
encode = lambda s : [stoi[char] for char in s] 
decode = lambda l : ''.join([itos[i] for i in l])

# train/test data
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]

# data loading
def get_batches(split): 
    data = train_data if split == 'train' else val_data
    ix = torch.randint((len(data) - block_size), (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

# average loss over eval_iters, less noisy
@torch.no_grad()
def estimate_loss(): 
    out = {}
    model.eval()
    for split in ['train', 'val']: 
        losses = torch.zeros(eval_iters)
        for i in range(eval_iters): 
            X, Y = get_batches(split)
            logits, loss = model(X, Y)
            losses[i] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# Bigram Language Model
class BigramLanguageModel(nn.Module): 
    def __init__(self): 
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None): 
        # both idx and targets are of dimension (B, T) tensors
        tok_emb = self.token_embedding_table(idx) # (B, T, n_embd)
        logits = self.lm_head(tok_emb) # (B, T, vocab_size)

        if targets is None: 
            loss = None
        else:  
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss
    
    def generate(self, idx, max_new_tokens): 
        # idx is (B, T) array of indices in the current context 
        for _ in range(max_new_tokens): 
            logits, loss = self(idx) # get the predictions
            logits = logits[:, -1, :] # grabs the last element in the T (time) dimension, becomes (B, C)
            probs = F.softmax(logits, dim=-1) # (B, C)
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx
    
model = BigramLanguageModel(vocab_size)
m = model.to(device)

# create an optimizer 
optimizer = torch.optim.AdamW(m.parameters(), learning_rate)

for iter in range(max_iters): 
    # after every eval_interval steps print the average loss of the model 
    if iter % eval_interval == 0: 
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    xb, yb = get_batches("train")

    # evaluate the losss
    logits, loss = m(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# generate from the model
idx = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(idx, max_new_tokens=500)[0].tolist()))
