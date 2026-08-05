import torch
from torch import nn
from src.utils import get_device, load_config
from src.models.build import build_model
import numpy as np
from src.evaluate import evaluate
from torch.utils.data import DataLoader
from src.data.dataset import NEUDataset as dataset
from pathlib import Path
from src.data.transforms import train_transform, eval_transform
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/baseline.yaml"),
    )
    return parser.parse_args()


def train(dataloader, model, loss_fn, optimizer, device):
    size = len(dataloader.dataset)
    train_loss, correct = 0, 0
    num_batches = len(dataloader)
    
    model.train()
    for batch, (x, y) in enumerate(dataloader):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        pred = model(x)
        
        loss = loss_fn(pred, y)
        
        train_loss += loss.item()
        correct += (pred.argmax(1) == y).type(torch.float).sum().item()
        
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
    train_loss /= num_batches
    correct /= size
    accuracy = 100 * correct
    
    print(f"Train error: \n Accuracy: {accuracy:>0.1f}%, Avg loss: {train_loss:>8f} \n")
    
    return accuracy, train_loss


def main():
    args = parse_args()
    cfg = load_config(args.config)
    device = get_device()

    model = build_model(device=device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam([
        {"params" : model.layer4.parameters(), "lr" : 1e-4},
        {"params" : model.fc.parameters(), "lr" : float(cfg['train']['lr'])},
    ])
    dataset_root = Path(cfg['dataset']['root'])
    batch_size = cfg['train']['batch_size']
    shuffle = cfg['train']['shuffle']
    train_dataloader = DataLoader(dataset(root_dir=Path(dataset_root / "train"), transform=train_transform), batch_size=batch_size, shuffle=shuffle)
    eval_dataloader = DataLoader(dataset(root_dir=Path(dataset_root / "validation"), transform=eval_transform), batch_size=batch_size, shuffle=shuffle)

    num_epochs = cfg['train']['epochs']

    scores = np.zeros((num_epochs, 2))
    prep_train_scores = scores.copy()
    prep_test_scores = scores.copy()

    for t in range(num_epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        prep_train_scores[t] = train(train_dataloader, model, loss_fn, optimizer, device)
        prep_test_scores[t] = evaluate(eval_dataloader, model, loss_fn, device)
    print("Done!")


if __name__ == "__main__":
    main()

