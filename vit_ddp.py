import datasets
import transformers

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
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torch.optim import SGD
import torch

# accelerate
import evaluate
from accelerate.utils import set_seed
from accelerate import Accelerator
from tqdm import tqdm

import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_name", type=str, default="nelorth/oxford-flowers")
    parser.add_argument(
        "--model_name", type=str, default="google/vit-base-patch16-224-in21k")
    parser.add_argument(
        "--resolution", type=int, default=512)
    parser.add_argument(
        "--batch_size", type=int, default=16)
    parser.add_argument(
        "--num_steps", type=int, default=500)
    parser.add_argument(
        "--save_dir", type=str, default="./saved/")
    parser.add_argument(
        "--epochs", type=int, default=9)                   # max epochs : must larger than steps
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    return args 


def main():
    
    args = parse_args()
    set_seed(args.seed)
    metric = evaluate.load("accuracy")
    
    accelerator = Accelerator()
    # device = accelerator.device

    # To have only one message (and not 8) per logs of Transformers or Datasets, we set the logging verbosity
    # to INFO for the main process only.
    if accelerator.is_main_process:
        datasets.utils.logging.set_verbosity_info()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()    
    

    # load dataset
    dataset = load_dataset(args.dataset_name, cache_dir='./dataset')
    
    if args.dataset_name == 'cifar10' :
        dataset = dataset.rename_column("img", "image")
        
    if args.dataset_name == 'cifar100' :
        dataset = dataset.rename_column("img", "image")
        dataset = dataset.rename_column("fine_label", "label")

    labels = dataset["train"].features["label"].names
    label2id, id2label = dict(), dict()

    for i, label in enumerate(labels):
        label2id[label] = i
        id2label[i] = label
        
    # model & processor
    # model_name = 'google/vit-base-patch16-224-in21k'
    processor = ViTImageProcessor.from_pretrained(args.model_name, cache_dir='./processor')
    model = ViTForImageClassification.from_pretrained(
        args.model_name,
        num_labels=len(id2label),
        id2label=id2label,
        label2id=label2id,
        cache_dir='./model'
    )

    # transforms : train_transforms <> val_transforms
    normalize = Normalize(mean=processor.image_mean, std=processor.image_std)

    """
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
    """
    train_transforms = Compose(
        [
            RandomResizedCrop((args.resolution, args.resolution)),
            RandomHorizontalFlip(),
            ToTensor(),
            normalize,
        ]
    )

    val_transforms = Compose(
        [
            Resize((args.resolution, args.resolution)),
            CenterCrop((args.resolution, args.resolution)),
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
    
    # epochs = 7
    # batch_size = 32
    # num_steps =  2500

    # dataset - dataloader
    # splits = dataset['train'].train_test_split(test_size=0.1)
    train_dataset = dataset['train']
    # val_dataset = splits['test']
    test_dataset = dataset["test"]

    train_dataset.set_transform(preprocess_train)
    test_dataset.set_transform(preprocess_val)

    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size)
    test_dataloader  = DataLoader(test_dataset, batch_size=args.batch_size)
    
    max_epochs = args.epochs

    # config
    optimizer = SGD(model.parameters(), lr=0.01, momentum=0.9)
    scheduler = CosineAnnealingLR(optimizer, T_max=max_epochs, eta_min=0)
    

    model, train_dataloader, test_dataloader, optimizer, scheduler= accelerator.prepare(
        model, train_dataloader, test_dataloader, optimizer, scheduler
    )
    
    step = 0  
    total_loss = 0  
    for i in range(max_epochs) :
        
        # print(f"train : epoch {i}")
        # accelerator.print(f"step {i}")
        model.train()
        # https://github.com/huggingface/accelerate/issues/2029
        for batch in tqdm(train_dataloader, disable=(not accelerator.is_local_main_process)):
            
            image = batch["image"] # .to(device)
            labels = batch["label"] # .to(device)
            
            outputs = model(image, labels = labels, interpolate_pos_encoding=True)
            loss = outputs.loss
            total_loss += loss

            accelerator.backward(loss)
            # loss.backward()

            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            
            step += 1
            
            # eval
            print_step = args.num_steps/5
            
            if step % print_step == 0:

                accelerator.print(f"\n   val : ")
                model.eval()
                for batch in test_dataloader :
                
                    image = batch["image"] # .to(device)
                    labels = batch["label"] # .to(device)
                
                    with torch.no_grad():
                        
                        outputs = model(image, labels = labels, interpolate_pos_encoding=True)
                        predictions = outputs.logits.argmax(dim=-1)
                        
                        predictions, references = accelerator.gather_for_metrics((predictions, batch["label"]))
                        metric.add_batch(
                            predictions=predictions,
                            references=references,
                        )              
                    
                eval_metric = metric.compute()
                
                accelerator.print(f"step : {step}, Training Loss : {total_loss/print_step} acc : {eval_metric}")
                total_loss = 0  # total_loss init

            # end
            if step == args.num_steps:
                break
            
        if step == args.num_steps:
                
            accelerator.wait_for_everyone()
            unwrapped_model = accelerator.unwrap_model(model)
            unwrapped_model.save_pretrained(
            args.save_dir+args.model_name+"_D_"+args.dataset_name+"R_"+str(args.resolution), 
            is_main_process=accelerator.is_main_process, save_function=accelerator.save
            )
            
            break


if __name__ == "__main__":
    main()