from B_read import get_labels_and_trainlist

import torch

import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

import torch.utils.data as Data

import torch.optim as optim

import torchvision
import torchvision.transforms as transforms


Train_Blist, labels = get_labels_and_trainlist()
#print(Train_Blist.shape)
Batch_size = 10
LRate = 0.000001
Epochs_num = 100

class Detection_network(nn.Module):
    def __init__(self, input_num):
        super(Detection_network, self).__init__()

        self.linear1 = nn.Linear(input_num, 15)
        self.linear2 = nn.Linear(15,5)
        self.out = nn.Linear(5,2)
        
    def forward(self, x):
        x = self.linear1(x)
        #x = F.relu(x)
        x = torch.sigmoid(x)

        x = self.linear2(x)
        #x = F.relu(x)
        x = torch.sigmoid(x)

        x = self.out(x)

        return x

test_net = Detection_network(20)

dataset = Data.TensorDataset(Train_Blist*5, labels/100)   # 将训练数据的特征和标签组合
data_iter = Data.DataLoader(dataset, Batch_size, shuffle=True)   # 随机读取小批量
optimizer = optim.SGD(test_net.parameters(),lr = LRate)
loss_fn = nn.MSELoss()

for epoch in range(Epochs_num):
    for data_in, label in data_iter:
       preds = test_net(data_in)
       loss = loss_fn(preds, label)

       optimizer.zero_grad()
       loss.backward()
       optimizer.step()

    print('epoch:',epoch,'  loss:',loss.item())

sample = next(iter(dataset))
list, label = sample
#print(list.unsqueeze(0).shape)
print(test_net(list.unsqueeze(0)))
print(label)