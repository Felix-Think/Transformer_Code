import torch
import torch.nn as nn

#|%%--%%| <CQye183iqc|nDe7Hh3zrs>

class TokenAndPositionEmbedding(nn.Module):
    def __init__(self, vocab_size, embed_dim, max_length, device = 'cpu'):
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
    def forward(self, output):
        batch_size, seq_length = output.size()
        positions = torch.arange(0, seq_length).expand(batch_size, seq_length).to(self.device)
        output1 = self.word_embedding(output)
        output2 = self.position_embedding(positions)
        return output1 + output2 # batch_size, seq_length, embed_dim

#|%%--%%| <nDe7Hh3zrs|AHzCxWjtUC>

class TransformerDecoderBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim = embed_dim,
            num_heads = num_heads,
            batch_first = True # return shape is batch_size, seq_length, embed_dim
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
                      bias = True)
        )
        
        self.norm1 = nn.LayerNorm(normalized_shape = embed_dim, eps = 1e-6)
        self.norm2 = nn.LayerNorm(normalized_shape = embed_dim, eps = 1e-6)
        self.norm3 = nn.LayerNorm(normalized_shape = embed_dim, eps = 1e-6)
        self.dropout1 = nn.Dropout(p = dropout)
        self.dropout2 = nn.Dropout(p = dropout)
        self.dropout3 = nn.Dropout(p = dropout)
    def forward(self, x, enc_output, src_mask, tgt_mask):
        attn_output, _ = self.attn(x, x, x, attn_mask = tgt_mask)
        attn_output = self.dropout1(attn_output)
        out1 = self.norm1(x + attn_output)

        attn_output, _ = self.cross_attn(out1, enc_output, enc_output, attn_mask = src_mask) # out1 is query or is input of decode, enc_output is key and value of encoder
        attn_output = self.dropout2(attn_output)
        out2 = self.norm2(out1 + attn_output)

        ffn_output = self.ffn(out2)
        ffn_output = self.dropout3(ffn_output)
        out3 = self.norm3(out2 + ffn_output)
        return out3

#|%%--%%| <AHzCxWjtUC|x1yNGL9k4t>

class TransformerDecoder(nn.Module):
    def __init__(self, tgt_vocab_size, embed_dim, max_length, num_layers, num_heads, ff_dim, dropout= 0.1, device = 'cpu'):
        super().__init__()
        self.device = device
        self.embedding = TokenAndPositionEmbedding(
            tgt_vocab_size, embed_dim, max_length, device
        )
        self.layers = nn.ModuleList(
            [
                TransformerDecoderBlock(
                    embed_dim, num_heads, ff_dim, dropout
                )for _ in range(num_layers)
            ]
        )
    def forward(self, x, enc_output, src_mask, tgt_mask):
        output = self.embedding(x)
        # Because in this case we don't have encoder, so we must use the same input for query, key, and value
        enc_output = self.embedding(enc_output)
        for layer in self.layers:
            output = layer(output, enc_output, src_mask, tgt_mask) # output is query or is input of decode, enc_output is key and value of encoder
        return output


#|%%--%%| <x1yNGL9k4t|H00HAdNP7T>


tgt_vocab_size = 2000
src_vocab_size = 1000
max_length = 100
batch_size = 128
embed_dim = 200
ff_dim = 256
num_heads = 4
num_layers = 2

#|%%--%%| <H00HAdNP7T|h6d5bBzYv4>

tgt_seq_len = 100
tgt_mask = torch.ones(tgt_seq_len, tgt_seq_len)
tgt_mask 

#|%%--%%| <h6d5bBzYv4|AIfA7zmGuJ>

#tgt_mask = tgt_mask.triu(diagonal = 1) # diagonal = 1 means we exclude the diagonal and below it
# But in this case we trust use triu
tgt_mask = torch.triu(tgt_mask)
tgt_mask

#|%%--%%| <AIfA7zmGuJ|YJvrMDJrBZ>

tgt_mask = tgt_mask.transpose(0, 1) # 0, 1 means we transpose the first and second dimension
tgt_mask

#|%%--%%| <YJvrMDJrBZ|dIy83DU1UT>

tgt_mask = tgt_mask.masked_fill(tgt_mask == 0, float('-inf'))
tgt_mask

#|%%--%%| <dIy83DU1UT|3LehVG5EG0>

tgt_mask = tgt_mask.masked_fill(tgt_mask == 1, 0)
tgt_mask
src_mask = tgt_mask
#|%%--%%| <3LehVG5EG0|uJFEGlQhwT>


model = TransformerDecoder(tgt_vocab_size, embed_dim, max_length, num_layers, num_heads, ff_dim)
src = torch.randint(
    high = 2,
    size = (batch_size, max_length),
    dtype = torch.int64
)

tgt = torch.randint(
    high = 2,
    size = (batch_size, max_length),
    dtype = torch.int64
)

#|%%--%%| <uJFEGlQhwT|8Vz1ewKnyY>

prediction = model(src, tgt, src_mask, tgt_mask)
prediction.shape # batch_size, seq_length, embed_dim

