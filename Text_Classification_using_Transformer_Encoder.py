from datasets import load_dataset
ds = load_dataset('thainq107/ntc-scv')
#|%%--%%| <8pLddrhI1Z|y1yldBG8YZ>

ds

#|%%--%%| <y1yldBG8YZ|gwf2GahCVn>
r"""°°°
# Preprocessing
°°°"""
#|%%--%%| <gwf2GahCVn|RmajXKYgSM>

import re
import string

def preprocess_text(text):
    # remove URLs https://www.
    url_pattern = re.compile(r'https?://\s+\wwww\.\s+') # https://www.
    text = url_pattern.sub(r" ", text) # replace with space

    # remove HTML Tags: <>
    html_pattern = re.compile(r'<[^<>]+>') 
    text = html_pattern.sub(" ", text)

    # remove puncs and digits
    replace_chars = list(string.punctuation + string.digits)
    for char in replace_chars:
        text = text.replace(char, " ")

    # remove emoji
    emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
        u"\U0001F1F2-\U0001F1F4"  # Macau flag
        u"\U0001F1E6-\U0001F1FF"  # flags
        u"\U0001F600-\U0001F64F"
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        u"\U0001f926-\U0001f937"
        u"\U0001F1F2"
        u"\U0001F1F4"
        u"\U0001F620"
        u"\u200d"
        u"\u2640-\u2642"
        "]+", flags=re.UNICODE)
    text = emoji_pattern.sub(r" ", text) # replace with space

    # normalize whitespace
    text = " ".join(text.split()) # replace multiple whitespaces with single whitespace

    # lowercasing
    text = text.lower()
    return text




#|%%--%%| <RmajXKYgSM|AFPMp0gHPp>
r"""°°°
# Representation
°°°"""
#|%%--%%| <AFPMp0gHPp|bbbgg3lc2x>

def yield_token(sentences, tokenizer):
    for sentence in sentences: # iterate over sentences
        yield tokenizer(sentence) 


#|%%--%%| <bbbgg3lc2x|6OvL0aAcJZ>

import torchtext
from torchtext.data.utils import get_tokenizer

tokenizer = get_tokenizer('basic_english')
sentences = ["Hello, how are you?", "I am fine, thank you."]
tokens = list(yield_token(sentences, tokenizer))
tokens
#|%%--%%| <6OvL0aAcJZ|xpCq3MzoAL>

#build vocabulary
from torchtext.vocab import build_vocab_from_iterator

vocab_size = 10000
vocabulary = build_vocab_from_iterator(
                yield_token(ds['train']['preprocessed_sentence'], tokenizer),
                max_tokens = vocab_size,
                specials = ['<unk>', '<pad>']
                )
vocabulary.set_default_index(vocabulary['<unk>'])

#|%%--%%| <xpCq3MzoAL|guDETdKB4s>

from torchtext.data.functional import to_map_style_dataset

def prepare_dataset(df):
    # create iterator for dataset: (sentence, label)
    for row in df:
        sentence = row['preprocessed_sentence']
        encoded_sentence = vocabulary(tokenizer(sentence))
        label = row['label']
        yield encoded_sentence, label

train_dataset = prepare_dataset(ds['train'])
train_dataset = to_map_style_dataset(train_dataset)

valid_dataset = prepare_dataset(ds['valid'])
valid_dataset = to_map_style_dataset(valid_dataset)

test_dataset = prepare_dataset(ds['test'])
test_dataset = to_map_style_dataset(test_dataset)

#|%%--%%| <guDETdKB4s|QkkX6uJ2m9>

for row in ds['train']:
    print(vocabulary(tokenizer(row['preprocessed_sentence'])))
    break

#|%%--%%| <QkkX6uJ2m9|jAEIdYirDe>

import torch

seq_length = 100

def collate_batch(batch):
    # create inputs, offsets, labels for batch
    sentences, labels = list(zip(*batch))
    encoded_sentences = [
        sentence+([0]* (seq_length-len(sentence))) if len(sentence) < seq_length else sentence[:seq_length]
        for sentence in sentences
    ]

    encoded_sentences = torch.tensor(encoded_sentences, dtype=torch.int64)
    labels = torch.tensor(labels)

    return encoded_sentences, labels
#|%%--%%| <jAEIdYirDe|v4hDWV1dEq>

from torch.utils.data import DataLoader

batch_size = 128

train_dataloader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    collate_fn=collate_batch
)
valid_dataloader = DataLoader(
    valid_dataset,
    batch_size=batch_size,
    shuffle=False,
    collate_fn=collate_batch
)

test_dataloader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,
    collate_fn=collate_batch
)
#|%%--%%| <v4hDWV1dEq|AJeLJ57OPn>

next(iter(train_dataloader))

#|%%--%%| <AJeLJ57OPn|Gqf89Ymda4>

len(train_dataloader)

#|%%--%%| <Gqf89Ymda4|d0QOpql7DD>

encoded_sentences, labels = next(iter(train_dataloader))

#|%%--%%| <d0QOpql7DD|CLMp9OevCS>

encoded_sentences.shape

#|%%--%%| <CLMp9OevCS|GP7mwMQXBY>

labels.shape

#|%%--%%| <GP7mwMQXBY|dXvH9YPY72>

import time

def train_epoch(model, optimizer, criterion, train_dataloader, device, epoch=0, log_interval=50):
    model.train()
    total_acc, total_count = 0, 0
    losses = []
    start_time = time.time()

    for idx, (inputs, labels) in enumerate(train_dataloader):
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        predictions = model(inputs)

        # compute loss
        loss = criterion(predictions, labels)
        losses.append(loss.item())

        # backward
        loss.backward()
        optimizer.step()
        total_acc += (predictions.argmax(1) == labels).sum().item()
        total_count += labels.size(0)
        if idx % log_interval == 0 and idx > 0:
            elapsed = time.time() - start_time
            print(
                "| epoch {:3d} | {:5d}/{:5d} batches "
                "| accuracy {:8.3f}".format(
                    epoch, idx, len(train_dataloader), total_acc / total_count
                )
            )
            total_acc, total_count = 0, 0
            start_time = time.time()

    epoch_acc = total_acc / total_count
    epoch_loss = sum(losses) / len(losses)
    return epoch_acc, epoch_loss


#|%%--%%| <dXvH9YPY72|bcm2xjRt1Z>


def evaluate_epoch(model, criterion, valid_dataloader, device):
    model.eval()
    total_acc, total_count = 0, 0
    losses = []

    with torch.no_grad():
        for idx, (inputs, labels) in enumerate(valid_dataloader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            predictions = model(inputs)

            loss = criterion(predictions, labels)
            losses.append(loss.item())

            total_acc += (predictions.argmax(1) == labels).sum().item()
            total_count += labels.size(0)

    epoch_acc = total_acc / total_count
    epoch_loss = sum(losses) / len(losses)
    return epoch_acc, epoch_loss
#|%%--%%| <bcm2xjRt1Z|egYIgjJG3j>

def train(model, model_name, save_model, optimizer, criterion, train_dataloader, valid_dataloader, num_epochs, device):
    train_accs, train_losses = [], []
    eval_accs, eval_losses = [], []
    best_loss_eval = 100
    times = []
    for epoch in range(1, num_epochs+1):
        epoch_start_time = time.time()
        # Training
        train_acc, train_loss = train_epoch(model, optimizer, criterion, train_dataloader, device, epoch)
        train_accs.append(train_acc)
        train_losses.append(train_loss)

        # Evaluation
        eval_acc, eval_loss = evaluate_epoch(model, criterion, valid_dataloader, device)
        eval_accs.append(eval_acc)
        eval_losses.append(eval_loss)

        # Save best model
        if eval_loss < best_loss_eval:
            torch.save(model.state_dict(), save_model + f'/{model_name}.pt')

        times.append(time.time() - epoch_start_time)
        # Print loss, acc end epoch
        print("-" * 59)
        print(
            "| End of epoch {:3d} | Time: {:5.2f}s | Train Accuracy {:8.3f} | Train Loss {:8.3f} "
            "| Valid Accuracy {:8.3f} | Valid Loss {:8.3f} ".format(
                epoch, time.time() - epoch_start_time, train_acc, train_loss, eval_acc, eval_loss
            )
        )
        print("-" * 59)

    # Load best model
    model.load_state_dict(torch.load(save_model + f'/{model_name}.pt', weights_only=True))
    model.eval()
    metrics = {
        'train_accuracy': train_accs,
        'train_loss': train_losses,
        'valid_accuracy': eval_accs,
        'valid_loss': eval_losses,
        'time': times
    }
    return model, metrics

#|%%--%%| <egYIgjJG3j|uYyfvNA701>

import matplotlib.pyplot as plt

def plot_result(num_epochs, train_accs, eval_accs, train_losses, eval_losses):
    epochs = list(range(num_epochs))
    fig, axs = plt.subplots(nrows = 1, ncols =2 , figsize = (12,6))
    axs[0].plot(epochs, train_accs, label = "Training")
    axs[0].plot(epochs, eval_accs, label = "Evaluation")
    axs[1].plot(epochs, train_losses, label = "Training")
    axs[1].plot(epochs, eval_losses, label = "Evaluation")
    axs[0].set_xlabel("Epochs")
    axs[1].set_xlabel("Epochs")
    axs[0].set_ylabel("Accuracy")
    axs[1].set_ylabel("Loss")
    plt.legend()




#|%%--%%| <uYyfvNA701|1dfpkvcdAW>
r"""°°°
# Modeling
°°°"""
#|%%--%%| <1dfpkvcdAW|qhvit5ssyA>

import torch.nn as nn
class TokenAndPositionEncoder(nn.Module):
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

#|%%--%%| <qhvit5ssyA|v3fYS5knT0>

class TransformerEncoderBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim = embed_dim,
            num_heads = num_heads,
            batch_first = True
        )
        self.ffn = nn.Sequential(
            nn.Linear(in_features = embed_dim, out_features = ff_dim, bias = True),
            nn.ReLU(),
            nn.Linear(in_features = ff_dim, out_features = embed_dim, bias = True)
        )
        self.layer_norm_1 = nn.LayerNorm(normalized_shape = embed_dim, eps = 1e-6)
        self.layer_norm_2 = nn.LayerNorm(normalized_shape = embed_dim, eps = 1e-6)
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
    def forward(self, query, key, value):
        attn_output, _ = self.attn(query, key, value)
        attn_output = self.dropout_1(attn_output)
        output_1 = self.layer_norm_1(query + attn_output)
        ffn_output = self.ffn(output_1)
        ffn_output = self.dropout_2(ffn_output)
        output_2 = self.layer_norm_2(output_1 + ffn_output)
        return output_2

#|%%--%%| <v3fYS5knT0|Ci5kIGeFhV>

class TransformerEncoderCls(nn.Module):
    def __init__(self, vocab_size, embed_dim, max_length, num_heads, ff_dim, num_layers, dropout = 0.1, device = 'cpu'):
        super().__init__()
        self.positions_embedding = TokenAndPositionEncoder(vocab_size, embed_dim, max_length, device)
        self.encoder_blocks = nn.ModuleList(
            [
                TransformerEncoderBlock(embed_dim, num_heads, ff_dim, dropout
                                        )for _ in range(num_layers)
            ]
        )
        self.avg_pooling = nn.AvgPool1d(kernel_size = max_length) 
        self.fc1 = nn.Linear(in_features = embed_dim, out_features = 20)
        self.fc2 = nn.Linear(in_features = 20, out_features = 2)
        self.dropout = nn.Dropout(dropout)
        self.device = device
    def forward(self, x):
        output = self.positions_embedding(x)
        for layer in self.encoder_blocks:
            output = layer(output, output, output)
        output = self.avg_pooling(output.permute(0,2,1)).squeeze() # output: [batch_size, embed_dim]
        output = self.fc1(output)
        output = self.dropout(output)
        output = self.fc2(output)
        return output



#|%%--%%| <Ci5kIGeFhV|Z6AddRnuLJ>

vocab_size = 10000
max_length = 100
embed_dim = 200
num_layers = 2
num_heads = 4
ff_dim = 128
dropout=0.1

model = TransformerEncoderCls(
    vocab_size, embed_dim, max_length, num_heads, ff_dim, num_layers, dropout
)


#|%%--%%| <Z6AddRnuLJ|DL64IfZK3J>

encoded_sentences.shape

#|%%--%%| <DL64IfZK3J|dxB4VchpXG>

predictions = model(encoded_sentences)
predictions.shape

#|%%--%%| <dxB4VchpXG|23she2M6g5>

predictions




#|%%--%%| <23she2M6g5|QUFEoYqM3a>
r"""°°°
# Training
°°°"""
#|%%--%%| <QUFEoYqM3a|biEJU3cNFC>

import os
import torch.optim as optim

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = TransformerEncoderCls(
    vocab_size, embed_dim, max_length, num_heads, ff_dim, num_layers, dropout, device
)
model.to(device)

criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.00005)

num_epochs = 50
save_model = './model'
os.makedirs(save_model, exist_ok = True)
model_name = 'model'

model, metrics = train(
    model, model_name, save_model, optimizer, criterion, train_dataloader, valid_dataloader, num_epochs, device
)

#|%%--%%| <biEJU3cNFC|TKEIFKKTeP>

test_acc, test_loss = evaluate_epoch(model, criterion, test_dataloader, device)
test_acc, test_loss

#|%%--%%| <TKEIFKKTeP|5RHSuljdZP>



