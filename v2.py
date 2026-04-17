import torch
import torch.nn as nn
from torch.nn import functional as F
from datetime import datetime

# hyperparameter
batch_size = 64#32 # how many independent sequences will we compute in parallel?
block_size = 256#128 # what is the maximum context length for prediction? 
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4#1e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# device = 'mps' if torch.backends.mps.is_available() else 'cpu'
eval_iters = 200
n_embd = 384#128
n_head = 6#4
n_layer = 6#4
dropout = 0.2
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

class Head(nn.Module): 
    """ one head of self-attention """

    def __init__(self, head_size):
        super().__init__()
        self.head_size = head_size
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x): 
        B, T, C = x.shape
        k = self.key(x) # (B, T, C)
        q = self.query(x) # (B, C, T)
        # compute affinities (attention scores)
        wei = k @ q.transpose(-2, -1) * (self.head_size ** -0.5) # (B, T, C) @ (B, C, T) --> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values 
        v = self.value(x) # (B, T, C)
        out = wei @ v # (B, T, T) @ (B, T, C) --> (B, T, C)
        return out
    
class MultiHeadAttention(nn.Module): 
    """ multiple heads of self-attention in parallel """

    def __init__(self, num_heads, head_size): 
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x): 
        out = torch.cat([h(x) for h in self.heads], dim=-1) # concatenating over the C (channel) dimension 
        out = self.dropout(self.proj(out))
        return out 
    
class FeedForward(nn.Module): 
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, n_embd): 
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4*n_embd),
            nn.ReLU(n_embd), 
            nn.Linear(4*n_embd, n_embd),
            nn.Dropout(dropout), 
        )
    
    def forward(self, x): 
        return self.net(x)
    
class Block(nn.Module): 
    """ Transformer block: communication followed by computation """

    def __init__(self, n_embd, n_head): 
        # n_embd: embedding dimensions, n_head: number of heads we'd like
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x): 
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

# Bigram Language Model
class BigramLanguageModel(nn.Module): 
    def __init__(self): 
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd) # final layernorm
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None): 
        B, T = idx.shape

        # both idx and targets are of dimension (B, T) tensors
        tok_emb = self.token_embedding_table(idx) # (B, T, n_embd)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T, C)
        #  x is not just the token identities but also the positions of where the tokens occur
        x = tok_emb + pos_emb # (B, T, C)
        x = self.blocks(x) # (B, T, C)
        x = self.ln_f(x) # (B, T, C)
        logits = self.lm_head(x) # (B, T, vocab_size)

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
            idx_cond = idx[:, -block_size:] # crop idx to the last block_size tokens
            logits, loss = self(idx_cond) # get the predictions
            logits = logits[:, -1, :] # grabs the last element in the T (time) dimension, becomes (B, C)
            probs = F.softmax(logits, dim=-1) # (B, C)
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx
    
model = BigramLanguageModel()
m = model.to(device)

# create an optimizer 
optimizer = torch.optim.AdamW(m.parameters(), learning_rate)


print('--------------------')
print(f'hyperparameters: batch size: {batch_size}, block size: {block_size}, embedding vector size: {n_embd}, number of heads (per multi-headed): {n_head}, number of layers: {n_layer}')
print('--------------------')

start_time = datetime.now()

min_val_loss = float('inf')

for iter in range(max_iters): 
    # after every eval_interval steps print the average loss of the model 
    if iter % eval_interval == 0 or iter == 4999: 
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

        # save the state of the model with the min val loss
        if losses['val'] < min_val_loss: 
            min_val_loss = losses['val']
            torch.save(m.state_dict(), 'weights.pt')
    
    xb, yb = get_batches("train")

    # evaluate the losss
    logits, loss = m(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

end_time = datetime.now()
total_seconds = int((end_time - start_time).total_seconds())
mins, secs = divmod(total_seconds, 60)
hrs, mins = divmod(mins, 60)
print(f'training length: {hrs}hr {mins}m {secs}s')

# get the best model state
m.load_state_dict(torch.load('weights.pt'))
print(f'best val loss: {min_val_loss:.4f}')

# generate from the model
idx = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(idx, max_new_tokens=500)[0].tolist()))
