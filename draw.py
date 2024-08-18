import torch.nn as nn
import torch
from torchvision import models
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from model import CNN_draw
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from collections import Counter
device=torch.device('cpu')
data=datasets.MNIST('data', train=False, download=True, transform=transforms.Compose([transforms.ToTensor()]))
data_loader=DataLoader(data, batch_size=64, shuffle=False)
image, label=next(iter(data_loader))
model=CNN_draw().to(device)
model.load_state_dict(torch.load('normal.pth', map_location=device))
model.eval()
print(model(image[0]))