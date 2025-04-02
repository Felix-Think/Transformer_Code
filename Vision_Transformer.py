"""°°°
# Load Datasets
°°°"""
#|%%--%%| <0jnkpjyetv|vglQG9K26M>

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
import torchvision.datasets as ImageFolder
import math
import os
import time


#|%%--%%| <vglQG9K26M|1eD6EdCnLO>

#!gdown 1vSevps_hV5zhVf6aWuN8X7dd-qSAIgcc
#!unzip ./flower_photos.zip

#|%%--%%| <1eD6EdCnLO|2g4LFY9Qoy>

#load data
data_path = './flower_photos'
dataset = ImageFolder.ImageFolder(root = data_path)
classes = dataset.classes
num_classes = len(classes)
num_samples = len(dataset)

#split data
VALID_RATIO, TRAIN_RATIO = 0.1, 0.8
n_train_samples = int(num_samples * TRAIN_RATIO)
n_val_samples = int(num_samples * VALID_RATIO)
n_test_samples = num_samples - n_train_samples - n_val_samples

train_dataset, val_dataset, test_dataset = random_split(
    dataset,
    [n_train_samples, n_val_samples, n_test_samples]
)

#|%%--%%| <2g4LFY9Qoy|a2puwQ6wQV>

#resize and convert to tensor

IMG_SIZE = 224

train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

test_trainsform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

#|%%--%%| <a2puwQ6wQV|07Scgi87ZD>

train_dataset.dataset.transform = train_transform
val_dataset.dataset.transform = test_trainsform
test_dataset.dataset.transform = test_trainsform
#|%%--%%| <07Scgi87ZD|HzEH3YtKlU>

# DATALOADERS

BATCH_SIZE = 64

train_loader = DataLoader(
    train_dataset,
    batch_size = BATCH_SIZE,
    shuffle = True
)

val_loader = DataLoader(
    val_dataset,
    batch_size = BATCH_SIZE
)

test_loader = DataLoader(
    test_dataset,
    batch_size = BATCH_SIZE
)





#|%%--%%| <HzEH3YtKlU|9nNRaxoxCN>
r"""°°°
# TRAINING FROM SCRATCH
#|%%--%%| <9nNRaxoxCN|PlQuqoDKGq>
r"""°°°
## 4.1 Modeling
°°°"""
#|%%--%%| <PlQuqoDKGq|YOZnMA3lzv>

class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim = embed_dim,
            num_heads = num_heads,
            batch_first = True
        )
        self.ffn = nn.Sequential(
                nn.Linear(in_features = embed_dim, out_features = ff_dim),
                nn.ReLU(),
                nn.Linear(in_features = ff_dim, out_features = embed_dim)
        )
        self.layer_norm1 = nn.LayerNorm(normalized_shape = embed_dim, eps = 1e-6)
        self.layer_norm2 = nn.LayerNorm(normalized_shape = embed_dim, eps = 1e-6)
        self.dropout1 = nn.Dropout(p = dropout)
        self.dropout2 = nn.Dropout(p = dropout)

    def forward(self, query, key, val):
        attn_output, _ = self.attn(query, key, val) # self-attention
        attn_output = self.dropout1(attn_output)
        out1 = self.layer_norm1(query + attn_output) # add and norm
        ffn_output = self.ffn(out1) # feed forward
        out2 = self.layer_norm2(out1 + ffn_output) # add and norm
        return out2


#|%%--%%| <YOZnMA3lzv|Ft4Jj6dzmK>

class PathPositionEmbedding(nn.Module):
    def __init__(self, img_size = 224, embed_dim = 512, patch_size = 16, device = 'cpu'):
        super().__init__()
        self.device = device
        self.conv1 = nn.Conv2d(
            in_channels = 3,
            out_channels = embed_dim, 
            kernel_size = patch_size,
            stride = patch_size,
            bias = False
        )
        scale = embed_dim ** -0.5
        self.position_embedding = nn.Parameter(
            scale * torch.randn((img_size // patch_size) ** 2, embed_dim)
        )
    
    def forward(self, x):
        x = self.conv1(x)  # shape - [*, with, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape - [*, with, grid * grid]
        x = x.permute(0, 2, 1)  # shape - [*, grid * grid, with]
        x = x + self.position_embedding
        return x

#|%%--%%| <Ft4Jj6dzmK|8DbrvGA1PH>

class VisionTransformerCls(nn.Module):
    def __init__(self, img_size, embed_dim, num_heads, ff_dim, dropout = 0.1, patch_size = 16, num_classes = 10, device = 'cpu'):
        super().__init__()
        self.device = device
        self.position_embedding = PathPositionEmbedding(
            img_size = img_size,
            embed_dim = embed_dim,
            patch_size = patch_size,
            device = device
        )
        self.transformer_layer = TransformerEncoder(
            embed_dim = embed_dim,
            num_heads = num_heads,
            ff_dim = ff_dim,
            dropout = dropout
        )
        #self.pooling = nn.AvgPool1d(kernel_size = max_length)
        self.fc1 = nn.Linear(in_features = embed_dim, out_features = 20)
        self.fc2 = nn.Linear(in_features = 20, out_features = num_classes)
        self.dropout = nn.Dropout(dropout)
        self.ReLu = nn.ReLU()
    def forward(self, x):
        output = self.position_embedding(x)
        out_put = self.transformer_layer(output, output, output)
        out_put = output[:, 0, :]
        out_put = self.dropout(out_put)
        out_put = self.fc1(out_put)
        out_put = self.ReLu(out_put)
        out_put = self.dropout(out_put)
        out_put = self.fc2(out_put)
        return out_put




#|%%--%%| <8DbrvGA1PH|JGypu2sPgC>
r"""°°°
## 4.2  Training
°°°"""
#|%%--%%| <JGypu2sPgC|aKwCbt0pwZ>

def train_epoch(model, optimizer, criterion, data_loader, device, epoch = 0, log_interval = 50):
    model.train()
    total_acc, total_count = 0, 0
    losses = []
    times = []
    start_time = time.time()

    for idx, (images, labels) in enumerate(data_loader):
        # dua vao device
        images, labels = images.to(device), labels.to(device)

        # zero gradients
        optimizer.zero_grad()

        # predict
        predictions = model(images)

        # tinh loss
        loss = criterion(predictions, labels)
        losses.append(loss.item())

        # backpropagation
        loss.backward()
        optimizer.step()

        total_acc += (predictions.argmax(1) == labels).sum().item()
        total_count += labels.size(0) # so luong anh

        if idx % log_interval == 0 and idx > 0:
            elapsed = time.time() - start_time
            print("| epoch {:3d} | {:5d} / {:5d} batches |"
                  "| accuracy: {:8.3f}% |".format(epoch, idx, len(data_loader), total_acc / total_count * 100))
            total_acc, total_count = 0,0
            start_time = time.time()
    epoch_acc = total_acc / total_count * 100
    epoch_loss = sum(losses) / len(losses)
    return epoch_acc, epoch_loss

def evaluate(model, criterion, data_loader, device):
    model.eval()
    total_acc, total_count = 0, 0
    losses = []
    with torch.no_grad():
        for idx, (images, labels) in enumerate(data_loader):
            images, labels = images.to(device), labels.to(device)
            predictions = model(images)
            loss = criterion(predictions, labels)
            losses.append(loss.item())
            total_acc += (predictions.argmax(1) == labels).sum().item()
            total_count += labels.size(0)
    epoch_acc = total_acc / total_count * 100
    epoch_loss = sum(losses) / len(losses)
    return epoch_acc, epoch_loss

def train_model(model, model_name, save_model, optimizer, criterion, train_loader, val_loader, num_epochs, device):
    train_accs, train_losses = [], [] # su dung de visualize
    val_accs, val_losses = [], [] # su dung de visualize
    best_loss_acc = 100 # gia tri khoi tao dung de luu nhung model tot
    times = []

    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        train_acc, train_loss = train_epoch(model, optimizer, criterion, train_loader, device, epoch)
        train_accs.append(train_acc)
        train_losses.append(train_loss)
        # evaluate

        val_acc, val_loss = evaluate(model, criterion, val_loader, device)
        val_accs.append(val_acc)
        val_losses.append(val_loss)
        
        # save model
        if val_loss < best_loss_acc:
            torch.save(model.state_dict(), save_model + f'/{model_name}.pt')
        times.append(time.time() - epoch_start_time)
        print('-' * 59)
        # Print loss and acc end of epoch
        print("| End of epoch {:3d} | Time: {:5.2f}s | Train Accuracy: {:8.3f}% | Train Loss: {:8.3f} "
              "| Val Accuracy: {:8.3f}% | Val Loss: {:8.3f} ".format(epoch, time.time() - epoch_start_time, train_acc, train_loss, val_acc, val_loss))
        print('-' * 59) 
    # load best model
    model.load_state_dict(torch.load(save_model + f'/{model_name}.pt'))
    model.eval()
    metrics = {
        'train_acc': train_accs,
        'train_loss': train_losses,
        'val_acc': val_accs,
        'val_loss': val_losses,
        'time': times
    }
    return model, metrics

def plot_result(num_epochs, train_accs, train_losses, val_accs, val_losses):
    epochs = list(range(num_epochs))
    fig, axs = plt.subplots(nrows = 1, ncols = 2, figsize = (12, 6))
    axs[0].plot(epoch, train_accs, label = 'Train Accuracy')
    axs[0].plot(epoch, val_accs, label = 'Val Accuracy')
    axs[1].plot(epoch, train_losses, label = 'Train Loss')
    axs[1].plot(epoch, val_losses, label = 'Val Loss')
    axs[0].set_title('Accuracy')
    axs[1].set_title('Loss')
    axs[0].set_xlabel('Epochs')
    axs[1].set_xlabel('Epochs')
    axs[0].set_ylabel('Accuracy')
    axs[1].set_ylabel('Loss')
    plt.legend()


#|%%--%%| <aKwCbt0pwZ|tBBzFgiLb6>

img_size = 224
embed_dim = 512
num_heads = 4
ff_dim = 128
dropout = 0.1
num_classes = len(classes)
path_size = 16
device = 'cuda' if torch.cuda.is_available() else 'cpu'

#|%%--%%| <tBBzFgiLb6|mYP5R7KF8f>


model = VisionTransformerCls(
    img_size = 224, embed_dim = embed_dim, num_heads = num_heads, ff_dim = ff_dim, num_classes = num_classes, device = device
)

model = model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = 1e-4)

num_epochs = 50
save_model = './vit_flower'

os.makedirs(save_model, exist_ok = True)
model_name = 'vit_flower'

model, metric = train_model(
    model, model_name, save_model, optimizer, criterion, train_loader, val_loader, num_epochs, device
)




#|%%--%%| <mYP5R7KF8f|cBn4WLVyHK>
r"""°°°
# 5.0 Fine-tuning
°°°"""
#|%%--%%| <cBn4WLVyHK|BPyKbGy1dy>

from transformers import ViTForImageClassification
id2label = {idx : label for idx, label in enumerate(classes)}
label2id = {label : idx for idx, label in enumerate(classes)}
model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224-in21k', 
                                                  num_labels = num_classes,
                                                  id2label = id2label,
                                                  label2id = label2id
                                                  )
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = model.to(device)

#|%%--%%| <BPyKbGy1dy|m0ym3KZN3b>

!pip install -q datasets accelerate evaluate

#|%%--%%| <m0ym3KZN3b|6LKY9IJrfw>

import numpy as np
import evaluate

metric = evaluate.load('accuracy')
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis = 1)
    return metric.compute(predictions = predictions, references = labels)


#|%%--%%| <6LKY9IJrfw|Lf36jMuUeW>

from transformers import ViTImageProcessor

feature_extractor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224-in21k')


#|%%--%%| <Lf36jMuUeW|VG0YEWtRBK>

from transformers import TrainingArguments, Trainer

metric_name = "accuracy"

args = TrainingArguments(
    f"vit_flowers",
    save_strategy="epoch",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    num_train_epochs=10,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model=metric_name,
    logging_dir='logs',
    remove_unused_columns=False,
)

#|%%--%%| <VG0YEWtRBK|LTxTABBzvv>

import torch

def collate_fn(examples):
    # example => Tuple(image, label)
    pixel_values = torch.stack([example[0] for example in examples])
    labels = torch.tensor([example[1] for example in examples])
    return {"pixel_values": pixel_values, "labels": labels}


trainer = Trainer(
    model,
    args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=collate_fn,
    compute_metrics=compute_metrics,
    tokenizer=feature_extractor,
)
#|%%--%%| <LTxTABBzvv|h4Dh8nE5px>

import wandb
wandb.init(mode='disabled')

#|%%--%%| <h4Dh8nE5px|bJyPWACYwX>

trainer.train()


#|%%--%%| <bJyPWACYwX|sIXeuDe4gs>

outputs = trainer.predict(test_dataset)

#|%%--%%| <sIXeuDe4gs|gX6JLpvhCi>

outputs.metrics

