import torch
import torch.nn as nn

#|%%--%%| <W05UQKZ4mK|5EuSlc7r5x>

class TokenAndPositionEmbedding(nn.Module):
    def __init__(self, vocab_size, embed_dim, max_lengths, device = 'cpu'):
        super().__init__()
        self.device = device
        self.word_embedding = nn.Embedding(
            num_embeddings = vocab_size, # Corpus size or dictionary size
            embedding_dim = embed_dim # Embedding dimension for each word
        )
        self.pos_embedding = nn.Embedding(
            num_embeddings = max_lengths, # Maximum length of sequential
            embedding_dim = embed_dim # Embedding dimension for each position
        )
        
    def forward(self, output):
        N, seq_length = output.size() # N is batch size, seq_length is sequence length
        positions = torch.arange(0, seq_length).expand(N, seq_length).to(self.device) # N x seq_length
        output1 = self.word_embedding(output) # N x seq_length x embed_dim
        output2 = self.pos_embedding(positions) # N x seq_length x embed_dim
        return output1 + output2
        
#|%%--%%| <5EuSlc7r5x|NAtul6jfUD>

class TransformerEncoderBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim = embed_dim,
            num_heads = num_heads,
            batch_first = True
        )
        self.ffn = nn.Sequential(
            nn.Linear(in_features = embed_dim, # Input dimension
                      out_features = ff_dim, # Feed forward dimension
                      bias = True # Bias term 
                        ),
            nn.ReLU(), # activation Function
            nn.Linear(in_features = ff_dim , # Input dimension
                      out_features = embed_dim, # Feed forward dimension
                      bias = True # Bias term
                    )
        )
        self.norm1 = nn.LayerNorm(normalized_shape = embed_dim, eps = 1e-6)
        self.norm2 = nn.LayerNorm(normalized_shape = embed_dim, eps = 1e-6)
        self.dropout1 = nn.Dropout(p = dropout)
        self.dropout2 = nn.Dropout(p = dropout)
        
    def forward(self, query, key, value):
        attn_output, _ = self.attn(query, key, value)
        attn_output = self.dropout1(attn_output)
        out1 = self.norm1(query + attn_output) # we must add the residual connection
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output)
        out2 = self.norm2(out1 + ffn_output) # we must add the residual connection
        return out2

#|%%--%%| <NAtul6jfUD|PwiQILORlX>

class TransformerEncoder(nn.Module):
    def __init__(self, src_vocab_size, embed_dim, max_lengths, num_layers, num_heads, ff_dim, dropout = 0.1, device = 'cpu'):
        super().__init__()
        self.device = device
        self.embedding = TokenAndPositionEmbedding(
            src_vocab_size, embed_dim, max_lengths, device
        )
        self.layers = nn.ModuleList(
            [
                TransformerEncoderBlock(
                    embed_dim, num_heads, ff_dim, dropout
                ) for _ in range(num_layers)
            ]
        )


    def forward(self, output):
        output = self.embedding(output)
        for layer in self.layers:
            output = layer(output, output, output)
        return output
#|%%--%%| <PwiQILORlX|bIpHo4E0S5>

batch_size = 32
max_lengths = 100
src_vocab_size = 1000
embed_dim = 128
ff_dim = 1024
num_heads = 8
num_layers = 6


#|%%--%%| <bIpHo4E0S5|QsjaLe3l4R>

input = torch.randint(
    high = 2,
    size = (batch_size, max_lengths), # Batch size x Sequence length
    dtype = torch.int64
)
model = TransformerEncoder(src_vocab_size, embed_dim, max_lengths, num_layers, num_heads, ff_dim, dropout)
output = model(input)
output.shape # Batch size x Sequence length x Embedding dimension

