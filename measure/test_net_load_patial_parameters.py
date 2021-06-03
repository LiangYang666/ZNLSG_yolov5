import torch.nn as nn
import torch.nn.functional as F
import torch
import ipdb
import torchvision
import os
from collections import OrderedDict
class MyNet(nn.Module):
    def __init__(self):
        super(MyNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.bn1 = nn.BatchNorm2d(6)
        self.conv2 = nn.Conv2d(6, 12, kernel_size=3)
        self.relu = nn.ReLU(inplace=True)
        self.bn2 = nn.BatchNorm2d(12)
        self.conv3 = nn.Conv2d(12, 6, kernel_size=3)
        self.bn3 = nn.BatchNorm2d(6)
        self.fc1 = nn.Linear(149**2*6, 120)
        self.fc2 = nn.Linear(120, 3)

    def forward(self, x):
        x = self.bn1(self.conv1(x))
        x = self.relu(x)
        # x = self.pool1(x)

        x = self.bn2(self.conv2(x))
        x = self.relu(x)
        x = self.pool1(x)

        x = self.bn3(self.conv3(x))
        x = self.relu(x)
        x = self.pool1(x)
        # ipdb.set_trace()

        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class MyNet1(nn.Module):
    def __init__(self):
        super(MyNet1, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=3)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.seq_net1 = nn.Sequential(
            nn.Conv2d(6, 3, 3),
            nn.BatchNorm2d(3),
            nn.ReLU(),
            nn.Conv2d(3, 6, 3)
        )
        self.fc1 = nn.Linear(((1024-(3-1))//2-2-2)**2*6, 120)
        self.fc2 = nn.Linear(120, 2)
    def forward(self, x):
        x = self.pool1(self.conv1(x))
        # ipdb.set_trace()
        x = self.seq_net1(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        # x = torch.sigmoid(self.fc2(x))
        return x

if __name__ == "__main__":
    model = MyNet1()
    for n, p in model.named_parameters():
        print(n)

    for n in model.state_dict():
        print(n)
    '''
    torch.save(model.state_dict(), 'test_net.pt')

    pretrained_dict = torch.load('test_net.pt')
    # pretrained_dict.__len__() -> 15
    pretrained_dict_change = OrderedDict()
    for (k, v), k1 in zip(pretrained_dict.items(), model.state_dict().keys()):  # change all
        pretrained_dict_change[k1] = v

    pretrained_dict_change_partial = OrderedDict()
    for (k, v), k1 in zip(pretrained_dict.items(), list(model.state_dict().keys())[:10]):  # change partial
        print('out', k1, ' == ',  k)
        pretrained_dict_change_partial[k1] = v

    pretrained_dict_change_partial_name = OrderedDict()
    for i, ((k, v), k1) in enumerate(zip(pretrained_dict.items(), list(model.state_dict().keys()))):  # change partial
        if 'seq_net1' in k1:
            break
        print(f'out{i} ',  k1, ' == ',  k)
        pretrained_dict_change_partial_name[k1] = v
    pretrained_dict_change = {k1: v for (k, v), k1 in zip(pretrained_dict.items(), model.state_dict())}   # change relys on model.state-dict()
    model.load_state_dict(pretrained_dict_change)
    '''


    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    input = torch.randn(3, 3, 1024, 1024)
    model(input)
