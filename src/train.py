import torch
from torch import nn
from utils import get_device, load_config
from models.build import build_model
import numpy as np
from evaluate import evaluate
from torch.utils.data import DataLoader
from data.dataset import NEUDataset as dataset
from pathlib import Path
from data.transforms import train_transform, eval_transform


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


if __name__ == "__main__":
    cfg = load_config()
    device = get_device()

    model = build_model(device=device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam([
        {"params" : model.layer4.parameters(), "lr" : 1e-4},
        {"params" : model.fc.parameters(), "lr" : 1e-3},
    ])
    dataset_root = Path(cfg['dataset']['root'])
    train_dataloader = DataLoader(dataset(root_dir=Path(dataset_root / "train"), transform=train_transform))
    eval_dataloader = DataLoader(dataset(root_dir=Path(dataset_root / "validation"), transform=train_transform))

    num_epochs = cfg['train']['epochs']

    scores = np.zeros((num_epochs, 2))
    prep_train_scores = scores.copy()
    prep_test_scores = scores.copy()

    for t in range(num_epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        prep_train_scores[t] = train(train_dataloader, model, loss_fn, optimizer, device)
        prep_test_scores[t] = evaluate(eval_dataloader, model, loss_fn, device)
    print("Done!")

