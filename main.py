import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from model import CNN
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from collections import Counter
device=torch.device('cpu')
data=datasets.MNIST('data', train=True, download=True, transform=transforms.Compose([transforms.ToTensor()]))
data_loader=DataLoader(data, batch_size=64, shuffle=True)

model=CNN().to(device)
optimizer=optim.Adam(model.parameters(), lr=0.01)
criterion=nn.CrossEntropyLoss()
for epoch in range(10):
    for batch, (x, y) in enumerate(data_loader):
        x=x.to(device)
        y=y.to(device)
        optimizer.zero_grad()
        output=model(x)
        loss=criterion(output, y)
        loss.backward()
        optimizer.step()
    print(f'Epoch: {epoch}, Batch: {batch}, Loss: {loss.item()}')

test_data=datasets.MNIST('data', train=False, download=True, transform=transforms.Compose([transforms.ToTensor()]))
labels = np.array([label for _, label in test_data])
counts = Counter(labels)
test_loader=DataLoader(test_data, batch_size=64, shuffle=False)
correct=0
fault=np.zeros(10)
total=0

with torch.no_grad():
    for x, y in test_loader:

        x=x.to(device)
        y=y.to(device)
        output=model(x)
        _, predicted=output.max(1)
        total+=y.size(0)
        for i in range(len(predicted)):
            if predicted[i]!=y[i]:
                fault[y[i]]+=1

        correct+=(predicted==y).sum().item()
print(f'Accuracy: {correct/total}')
for i in range(10):
    print(f'Error of {i}: {fault[i]/ counts[i]}')

torch.save(model.state_dict(), 'normal.pth')