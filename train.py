import numpy as np

from tqdm import tqdm, trange

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader

import torchvision.transforms as transforms
import torchvision.datasets as data

np.random.seed(0)
torch.manual_seed(0)

import argparse

from VITModel import MyViT

import os

def parse_arguments():
    parser = argparse.ArgumentParser(description="ONNX Model Framework")
    parser.add_argument("--resume", choices=["true", "false"], default="false", help="Resume training from saved weights")
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_arguments()
    resume = args.resume

    if not os.path.exists('/work/mohsin/transfor_flower102/weights'):
        os.makedirs('/work/mohsin/transfor_flower102/weights')



    transform=transforms.Compose([transforms.Resize([256,256]),transforms.ToTensor()])

    train_set = data.Flowers102(root='./../datasets', split = 'test', download=True, transform=transform)
    train_loader = DataLoader(train_set, shuffle=True, batch_size=32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device: ", device, f"({torch.cuda.get_device_name(device)})" if torch.cuda.is_available() else "")
    model = MyViT((3, 256, 256), n_patches=16, n_blocks=12, hidden_d=768, n_heads=12, out_d=102).to(device)
    N_EPOCHS = 100
    LR = 0.00005

    if resume == "true":
        model.load_state_dict(torch.load('/work/mohsin/transfor_flower102/weights/oxford_normal_big_final.pth'))

    optimizer = Adam(model.parameters(), lr=LR)
    scheduler = StepLR(optimizer, step_size=15, gamma=0.1)

    criterion = CrossEntropyLoss()
    for epoch in trange(N_EPOCHS, desc="Training"):
        train_loss = 0.0
        correct, total = 0, 0
        acc_max = 0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1} in training", leave=False):
            x, y = batch
            x, y = x.to(device), y.to(device)
            y_hat = model(x)
            loss = criterion(y_hat, y)

            train_loss += loss.detach().cpu().item() / len(train_loader)
            correct += torch.sum(torch.argmax(y_hat, dim=1) == y).detach().cpu().item()
            total += len(x)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        scheduler.step()

        print(f"Epoch {epoch + 1}/{N_EPOCHS} loss: {train_loss:.2f} Accuracy: {correct / total * 100:.2f}%")
        acc_current = correct / total * 100

        if (acc_current > acc_max):
            acc_max = acc_current
            name = '/work/mohsin/transfor_flower102/weights/oxford_normal_big_max.pth'
            torch.save(model.state_dict(), name)

        torch.save(model.state_dict(), '/work/mohsin/transfor_flower102/weights/oxford_normal_big_final.pth')
    print("Training Done.")
