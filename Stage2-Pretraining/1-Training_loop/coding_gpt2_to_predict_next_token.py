# %% [markdown]
# Import libraries

# %%
import torch
import torch.nn as nn
import tiktoken

# %% [markdown]
# Confriguration considered

# %%
GPT_CONFIG_124M = {
    "vocab_size" : 50527,
    "context_length" : 768,
    "emb_dim" : 768,
    "n_heads" : 12,
    "n_layers" : 12,
    "drop_rate" : 0.1,
    "qkv_bias" : False
}

# %% [markdown]
# Layer normalization and Feed forward neural network class

# %%
class LayerNorm(nn.Module):
    def __init__(self,emb_dim):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim = True)
        variance = x.var(dim = -1, keepdim =True)
        norm_x = (x-mean) / torch.sqrt(variance + self.eps)
        return self.scale * norm_x + self.shift

class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 0.5*x*(1+torch.tanh(torch.sqrt(torch.tensor(2.0/torch.pi)) * 
        (x + 0.0044715*torch.pow(x,3))))


class FeedForward(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
            GELU(),
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"])
        )

    def forward(self, x):
        return self.layers(x)
    
    

# %% [markdown]
# Multi-Head attention class

# %%
class MultiHeadAttention(nn.Module):

    def __init__(self,d_in,d_out,context_length,dropout=0.5,num_heads=2 ,qkvbias = False):

        super().__init__()
        assert (d_out % num_heads == 0), \
            "d_out must be divisible by num_heads"
        
        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads

        self.W_query = nn.Linear(d_in,d_out,qkvbias)
        self.W_key = nn.Linear(d_in,d_out,qkvbias)
        self.W_value = nn.Linear(d_in,d_out,qkvbias)
        self.out_proj = nn.Linear(d_out,d_out)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer("mask", torch.tril(torch.ones(context_length, context_length),diagonal=1))

    def forward(self, x):

        b, num_tokens , d_in = x.shape
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        keys = keys.view(b,num_tokens,self.num_heads,self.head_dim)
        queries = queries.view(b,num_tokens,self.num_heads,self.head_dim)
        values = values.view(b,num_tokens,self.num_heads,self.head_dim)

        keys = keys.transpose(1,2)  # (b, num_heads, num_tokens, head_dim)
        queries = queries.transpose(1,2)  # (b, num_heads, num_tokens, head_dim)
        values = values.transpose(1,2)  # (b, num_heads, num_tokens, head_dim)

        attn_scores = queries @ keys.transpose(2,3)

        mask_bool = self.mask.bool()[:num_tokens, :num_tokens] ==0

        attn_scores = attn_scores.masked_fill_(mask_bool,-torch.inf)

        attn_weights = torch.softmax(attn_scores/keys.shape[-1]**0.5,dim =-1)
        attn_weights = self.dropout(attn_weights)

        context_vector = (attn_weights @ values ).transpose(1,2) #Shape: (b, num_tokens, num_heads, head_dim)

        context_vector = context_vector.contiguous().view(b,num_tokens,self.d_out)
        context_vector = self.out_proj(context_vector)

        return context_vector

# %% [markdown]
# Transformer class

# %%
class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.attn = MultiHeadAttention(
            d_in = cfg['emb_dim'],
            d_out = cfg['emb_dim'],
            context_length = cfg['context_length'],
            num_heads = cfg['n_heads'],
            dropout = cfg['drop_rate'],
            qkvbias = cfg['qkv_bias'])

        self.ffn = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg['emb_dim'])
        self.norm2 = LayerNorm(cfg['emb_dim'])
        self.dropout_shortcut = nn.Dropout(cfg['drop_rate'])

    def forward(self,x):
        # Shortcut connection for attention block
        shortcut = x
        x = self.norm1(x)
        x = self.attn(x)
        x = self.dropout_shortcut(x)
        x = x + shortcut

        # Shortcut connection for feedforward block
        shortcut =x
        x = self.norm2(x)
        x = self.ffn(x)
        x = self.dropout_shortcut(x)
        x = x + shortcut    

        return x


# %% [markdown]
# Full_GPT_archetecture_class

# %%
class GPTModel(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg['vocab_size'],cfg['emb_dim'])
        self.poc_emb = nn.Embedding(cfg['context_length'],cfg['emb_dim'])
        self.drop_emb = nn.Dropout(cfg['drop_rate'])

        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg['n_layers'])])
        
        self.final_norm = LayerNorm(cfg['emb_dim'])
        self.outhead = nn.Linear(cfg['emb_dim'], cfg['vocab_size'], bias = False)

    def forward(self,in_idx):
        batch_size,seq_length = in_idx.shape
        tok_embeds  =self.tok_emb(in_idx)
        pos_embeds = self.poc_emb(torch.arange(seq_length, device = in_idx.device))
        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.outhead(x)
        return logits


# %% [markdown]
# Generate next token

# %%
def generate_text_simple(model,idx,max_new_tokens,context_size):

    for _ in range(max_new_tokens):

        idx_cond = idx[:, -context_size:]
        # print(idx_cond)

        with torch.no_grad():
            logits = model(idx_cond)

        logits = logits[:,-1,:]

        probas = torch.softmax(logits,dim =-1) # (batch, vocab_size)

        idx_next = torch.argmax(probas, dim =-1, keepdim = True) # (batch, 1)

        idx = torch.cat((idx, idx_next), dim =1) # (batch, n_tokens+1)
    return idx

# %% [markdown]
# The softmax function is monotonic, meaning it preserves the order of its inputs when transformed into outputs. 
# 
# So, in practice, the softmax step isredundant since the position with the highest score in the softmax output tensor is the
# same position in the logit tensor. 
# 
# In other words, we could apply the torch.argmax function to the logits tensor directly and get identical results. 
# 
# 
# But implementing softmax will be useful when applying addtional additional sampling techniques w in the model where we modify the softmax outputs such that the model doesn't always select the most likely token, which introduces variability andcreativity in the generated text.