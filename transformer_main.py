import torch
import torch.nn as nn
import random
import time


#|%%--%%| <MGjqXj5zN0|oJCYSLNpM1>
r"""°°°
## Token Anh Position Embedding
°°°"""
#|%%--%%| <oJCYSLNpM1|5m9G4tYiJX>

class TokenAndPositionEmbedding(nn.Module):
    def __init__(self, vocab_size, embed_dim, max_length, device):
        super().__init__()
        self.device = device
        self.word_embedding = nn.Embedding(
            num_embeddings = vocab_size,
            embedding_dim = embed_dim
        )
        self.position_embedding = nn.Embedding(
            num_embeddings = max_length,
            embedding_dim = embed_dim
        )
    def forward(self, x):
        batch_size, seq_length = x.size()
        positions = torch.arange(0, seq_length).expand(batch_size, seq_length).to(self.device)
        output_1 = self.word_embedding(x)
        output_2 = self.position_embedding(positions)
        return output_1 + output_2

#|%%--%%| <5m9G4tYiJX|HTUPsMp9HC>

x = torch.tensor([[1, 2, 3], [4, 5, 6]])  # batch_size = 2, seq_length = 3

vocab_size = 100
embed_dim = 128
max_length = 3
device = 'cpu'

embedding = TokenAndPositionEmbedding(vocab_size, embed_dim, max_length, device)

embedding(x).shape  # torch.Size([2, 3, 128])


#|%%--%%| <HTUPsMp9HC|reCilEbNaf>

class TransformerEncoderBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout = 0.1, device = 'cpu'):
        super().__init__()
        self.device = device
        self.attn = nn.MultiheadAttention(
            embed_dim = embed_dim,
            num_heads = num_heads,
            batch_first = True
        )

        self.ffn = nn.Sequential(
            nn.Linear(in_features = embed_dim,
                      out_features = ff_dim,
                      bias = True
            ),
            nn.ReLU(),
            nn.Linear(in_features = ff_dim,
                      out_features = embed_dim,
                      bias = True)
        )
        self.layernorm_1 = nn.LayerNorm(normalized_shape = embed_dim, eps = 1e-6)
        self.layernorm_2 = nn.LayerNorm(normalized_shape = embed_dim, eps = 1e-6)
        self.dropout_1 = nn.Dropout(p = dropout)
        self.dropout_2 = nn.Dropout(p = dropout)

    def forward(self, query, key, value):
        attn_output, _ = self.attn(query, key, value) # self-attention
        attn_output = self.dropout_1(attn_output)
        output_1 = self.layernorm_1(query + attn_output) # residual connection
        ffn_output = self.ffn(output_1)
        ffn_output = self.dropout_2(ffn_output)
        output_2 = self.layernorm_2(output_1 + ffn_output) # residual connection
        return output_2

#|%%--%%| <reCilEbNaf|DcxigkF88H>

batch_size = 2
max_length = 5
embed_dim = 128
num_heads = 8
ff_dim = 256
dropout = 0.1
device = 'cpu'

encoder_block = TransformerEncoderBlock(embed_dim, num_heads, ff_dim, dropout, device)
embedding = TokenAndPositionEmbedding(vocab_size, embed_dim, max_length, device)
input = torch.randint(
    high = 2,
    size = (batch_size, max_length),
    dtype = torch.int64
)

embedded_input = embedding(input) # batch_size, max_length, embed_dim
output = encoder_block(embedded_input, embedded_input, embedded_input)
print(output.shape) # torch.Size([2, 5, 128])
#|%%--%%| <DcxigkF88H|xdGyeR6sa5>

class TransformerEncoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_heads, ff_dim, num_layers, max_length, dropout = 0.1, device = 'cpu'):
        super().__init__()
        self.device = device
        # Token and Position Embedding
        self.embedding = TokenAndPositionEmbedding(vocab_size, embedding_dim, max_length, device)
        # Transformer Encoder Blocks
        self.layers = nn.ModuleList(
            [
                TransformerEncoderBlock(embedding_dim, num_heads, ff_dim, dropout
                )for _ in range(num_layers)
            ] 
        )

    def forward(self, x):
        output = self.embedding(x)
        for layer in self.layers:
            output = layer(output, output, output)
        return output

#|%%--%%| <xdGyeR6sa5|NmxVLHzx3i>


batch_size = 128
max_length = 5
embed_dim = 128
num_heads = 8
ff_dim = 256
dropout = 0.1
num_layers = 4
device = 'cpu'

encoder = TransformerEncoder(vocab_size, embed_dim, num_heads, ff_dim, num_layers, max_length, dropout, device)
input = torch.randint(
    high = 2,
    size = (batch_size, max_length),
    dtype = torch.int64
)
encoder(input).shape # torch.Size([2, 5, 128])

#|%%--%%| <NmxVLHzx3i|ScOUp3K1Ia>

class TransformerDecoderBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim = embed_dim,
            num_heads = num_heads,
            batch_first = True
        )
        self.cross_attn = nn.MultiheadAttention(
            embed_dim = embed_dim,
            num_heads = num_heads,
            batch_first = True
        )
        self.ffn = nn.Sequential(
            nn.Linear(in_features = embed_dim,
                      out_features = ff_dim,
                      bias = True),
            nn.ReLU(),
            nn.Linear(in_features = ff_dim,
                      out_features = embed_dim,
                      bias= True)
        )
        self.layernorm_1 = nn.LayerNorm(normalized_shape = embed_dim, eps = 1e-6)
        self.layernorm_2 = nn.LayerNorm(normalized_shape = embed_dim, eps = 1e-6)
        self.layernorm_3 = nn.LayerNorm(normalized_shape = embed_dim, eps = 1e-6)
        self.dropout_1 = nn.Dropout(p = dropout)
        self.dropout_2 = nn.Dropout(p = dropout)
        self.dropout_3 = nn.Dropout(p = dropout)
    def forward(self, x, enc_output, src_mask, tgt_mask):
        attn_output, _ = self.attn(x, x, x, attn_mask = tgt_mask) # self-attention
        attn_output = self.dropout_1(attn_output)
        output_1 = self.layernorm_1(x + attn_output) # residual connection
        attn_output, _ = self.cross_attn(output_1, enc_output, enc_output, attn_mask = src_mask) # cross-attention
        attn_output = self.dropout_2(attn_output)
        output_2 = self.layernorm_2(output_1 + attn_output)
        ffn_output = self.ffn(output_2)
        ffn_output = self.dropout_3(ffn_output)
        output_3 = self.layernorm_3(output_2 + ffn_output) # residual connection
        return output_3

#|%%--%%| <ScOUp3K1Ia|nfcZpEVBSa>


class TransformerDecoder(nn.Module):
    def __init__(self,tgt_vocab_size,  embed_dim, max_length, num_heads, ff_dim, num_layers, dropout = 0.1, device = 'cpu'):
        super().__init__()
        self.embedding = TokenAndPositionEmbedding(tgt_vocab_size, embed_dim, max_length, device)
        self.layers = nn.ModuleList(
            [
                TransformerDecoderBlock(embed_dim, num_heads, ff_dim, dropout
                )for _ in range(num_layers)
            ]
        )
    def forward(self, x, enc_output, src_mask, tgt_mask):
        output = self.embedding(x)
        for layer in self.layers:
            output = layer(output, enc_output, src_mask, tgt_mask)
        return output

#|%%--%%| <nfcZpEVBSa|OLuIljcOQo>


class Transformer(nn.Module):
    def __init__(self, tgt_vocab_size, src_vocab_size, tgt_max_length, src_max_length, embed_dim, num_heads, ff_dim, num_layers, dropout = 0.1, device = 'cpu'):
        super().__init__()
        self.device = device
        self.encoder = TransformerEncoder(src_vocab_size, embed_dim, num_heads, ff_dim, num_layers, src_max_length, dropout, device)
        self.decoder = TransformerDecoder(tgt_vocab_size, embed_dim, tgt_max_length, num_heads, ff_dim, num_layers, dropout, device)
        self.fc = nn.Linear(in_features = embed_dim, out_features = tgt_vocab_size) 
    
    def generate_mask(self, src, tgt):
        src_mask = src.shape[1] # src shape: batch_size, src_max_length
        tgt_mask = tgt.shape[1]

        src_mask = torch.zeros((src_mask, src_mask), device = self.device).type(torch.bool)
        tgt_mask = (torch.triu(torch.ones(
            (tgt_mask, tgt_mask),
            device = self.device)
        ) == 1).transpose(0, 1)
        tgt_mask = tgt_mask.float().masked_fill(tgt_mask == 0, float('-inf')).masked_fill(tgt_mask == 1, float(0.0))
        return src_mask, tgt_mask

    def forward(self, src, tgt):
        src_mask, tgt_mask = self.generate_mask(src, tgt)
        enc_output = self.encoder(src)
        dec_output = self.decoder(tgt, enc_output, src_mask, tgt_mask)
        output = self.fc(dec_output)
        return output
#|%%--%%| <OLuIljcOQo|OBvlk9wb1A>

batch_size = 128
src_vocab_size = 1000
tgt_vocab_size = 2000
embed_dim = 200
tgt_max_length = 100
src_max_length = 100
num_layers = 2
num_heads = 4
ff_dim = 256

#|%%--%%| <OBvlk9wb1A|SEk85cTcc1>

model = Transformer(
    tgt_vocab_size, src_vocab_size, 
    tgt_max_length, src_max_length, 
    embed_dim, num_heads, 
    ff_dim, num_layers)

src = torch . randint (
    high = 2,
    size = (batch_size , src_max_length),
    dtype = torch.int64
)

tgt = torch.randint (
    high = 2,
    size = (batch_size, tgt_max_length),
    dtype = torch.int64
)
prediction = model ( src , tgt )
prediction.shape # batch_size x max_length x tgt_vocab_size

