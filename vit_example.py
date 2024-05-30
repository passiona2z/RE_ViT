from datasets import load_dataset
from transformers import ViTForImageClassification, ViTImageProcessor
from torchvision.transforms import (
    CenterCrop,
    Compose,
    Normalize,
    RandomHorizontalFlip,
    RandomResizedCrop,
    Resize,
    ToTensor
)

from torch.utils.data import DataLoader
from torch.optim import SGD
import torch

import evaluate
from tqdm import tqdm

metric = evaluate.load("accuracy")

# load dataset
dataset = load_dataset('cifar10', cache_dir='./dataset')
dataset = dataset.rename_column("img", "image")

print(dataset)

labels = dataset["train"].features["label"].names
label2id, id2label = dict(), dict()

for i, label in enumerate(labels):
    label2id[label] = i
    id2label[i] = label
    
# model & processor
model_name = 'google/vit-base-patch16-224-in21k'
processor = ViTImageProcessor.from_pretrained(model_name, cache_dir='./processor')
model = ViTForImageClassification.from_pretrained(
    model_name,
    num_labels=len(id2label),
    id2label=id2label,
    label2id=label2id,
    cache_dir='./model'
)

# transforms : train_transforms <> val_transforms
normalize = Normalize(mean=processor.image_mean, std=processor.image_std)

train_transforms = Compose(
        [
            RandomResizedCrop((processor.size['height'], processor.size['width'])),
            RandomHorizontalFlip(),
            ToTensor(),
            normalize,
        ]
    )

val_transforms = Compose(
        [
            Resize((processor.size['height'], processor.size['width'])),
            CenterCrop((processor.size['height'], processor.size['width'])),
            ToTensor(),
            normalize,
        ]
    )

def preprocess_train(example_batch):
    """Apply train_transforms across a batch."""
    example_batch["image"] = [
        train_transforms(image.convert("RGB")) for image in example_batch["image"]
    ]
    return example_batch

def preprocess_val(example_batch):
    """Apply val_transforms across a batch."""
    example_batch["image"] = [val_transforms(image.convert("RGB")) for image in example_batch["image"]]
    return example_batch

# dataset - dataloader
splits = dataset['train'].train_test_split(test_size=0.1)
train_dataset = splits['train']
val_dataset = splits['test']


train_dataset.set_transform(preprocess_train)
val_dataset.set_transform(preprocess_val)


train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=64)
val_dataloader  = DataLoader(val_dataset, batch_size=64)


# config
optimizer = SGD(model.parameters(), lr=0.01, momentum=0.9)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# train
model.to(device)


for i in range(3) :
    
    model.train()
    total_loss = 0
    for batch in tqdm(val_dataloader):
        
        image = batch["image"].to(device)
        labels = batch["label"].to(device)
        
        outputs = model(image, labels = labels)
        loss = outputs.loss
        total_loss += loss

        loss.backward()

        optimizer.step()
        optimizer.zero_grad()

    model.eval()
    for batch in val_dataloader :
        
        image = batch["image"].to(device)
        labels = batch["label"].to(device)
    
    with torch.no_grad():
           
        outputs = model(image, labels = labels)
        predictions = outputs.logits.argmax(dim=-1)
        
        metric.add_batch(
                predictions=predictions,
                references=batch["label"],
            )        

    eval_metric = metric.compute()
    
    print(f'epoch : {i+1}, Training Loss : {total_loss/len(train_dataloader)} acc : {eval_metric}')

