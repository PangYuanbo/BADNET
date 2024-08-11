import torch.nn
import torch.optim
import torch.nn.functional as F
class CNN (torch.nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 16,5,1)
        self.conv2 = torch.nn.Conv2d(16, 32, 5, 1)
        self.fc1 = torch.nn.Linear(4*4*32, 512)
        self.fc2 = torch.nn.Linear(512, 10)

    def forward(self, x):
        x=F.relu(self.conv1(x))
        x=F.avg_pool2d(x,2,2)
        x=F.relu(self.conv2(x))
        x=F.avg_pool2d(x,2,2)
        x=x.view(-1, 4*4*32)
        x=F.relu(self.fc1(x))
        x=self.fc2(x)
        return F.log_softmax(x, dim=1)
