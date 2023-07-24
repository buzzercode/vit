import numpy as np

from tqdm import tqdm

import torch
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader

import torchvision.transforms as transforms
import torchvision.datasets as data

np.random.seed(0)
torch.manual_seed(0)

from VITModel import MyViT

if __name__ == '__main__':

    transform=transforms.Compose([transforms.Resize([256,256]),transforms.ToTensor()])

    test_set = data.Flowers102(root='./../datasets', split = 'train', download=True, transform=transform)
    test_loader = DataLoader(test_set, shuffle=False, batch_size=32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device: ", device, f"({torch.cuda.get_device_name(device)})" if torch.cuda.is_available() else "")
    model = MyViT((3, 256, 256), n_patches=16, n_blocks=12, hidden_d=768, n_heads=12, out_d=102).to(device)

    model.load_state_dict(torch.load('weights/oxford_normal_big_final.pth'))
    criterion = CrossEntropyLoss()
    with torch.no_grad():
        correct, total = 0, 0
        test_loss = 0.0
        for batch in tqdm(test_loader, desc="Testing"):
            x, y = batch
            x, y = x.to(device), y.to(device)
            y_hat = model(x)
            loss = criterion(y_hat, y)
            test_loss += loss.detach().cpu().item() / len(test_loader)

            correct += torch.sum(torch.argmax(y_hat, dim=1) == y).detach().cpu().item()
            total += len(x)
        print(f"Test loss: {test_loss:.2f}")
        print(f"Test accuracy: {correct / total * 100:.2f}%")
