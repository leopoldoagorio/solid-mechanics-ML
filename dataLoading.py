"""Loading the csv with torch's data loader and using it for batch training
remember the csv structure was: echo "$Lx,$Ly,$Lz, $E, $nu, $p,$Ux,$Uy,$Uz" >> "$filename"
where Lx, Ly, Lz are the block's length, E and nu are material parameter,  p is the input pressure
 and Ux, Uy, Uz are the output compression """
import torch.nn as nn
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
import time

# Loading the data
class uniCompDataset(Dataset):
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file, header=None)
        self.root = csv_file
        self.len = self.data.shape[0]
    def __getitem__(self, index):
        return torch.tensor(self.data.iloc[index, :].values, dtype=torch.float)
    def __len__(self):
        return self.len

# Creating the model
class uniCompNet(nn.Module):
    def __init__(self):
        super(uniCompNet, self).__init__()
        self.fc1 = nn.Linear(3, 10)
        self.fc2 = nn.Linear(10, 10)
        self.fc3 = nn.Linear(10, 10)
        self.fc4 = nn.Linear(10, 10)
        self.fc5 = nn.Linear(10, 10)
        self.fc6 = nn.Linear(10, 10)
        self.fc7 = nn.Linear(10, 10)
        self.fc8 = nn.Linear(10, 10)
        self.fc9 = nn.Linear(10, 10)
        self.fc10 = nn.Linear(10, 10)
        self.fc11 = nn.Linear(10, 10)
        self.fc12 = nn.Linear(10, 10)
        self.fc13 = nn.Linear(10, 10)
        self.fc14 = nn.Linear(10, 10)
        self.fc15 = nn.Linear(10, 10)
        self.fc16 = nn.Linear(10, 10)
        self.fc17 = nn.Linear(10, 10)
        self.fc18 = nn.Linear(10, 10)
        self.fc19 = nn.Linear(10, 10)
        self.fc20 = nn.Linear(10, 10)
        self.fc21 = nn.Linear(10, 10)
        self.fc22 = nn.Linear(10, 10)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = F.relu(self.fc6(x))
        x = F.relu(self.fc7(x))
        x = F.relu(self.fc8(x))
        x = F.relu(self.fc9(x))
        x = F.relu(self.fc10(x))
        x = F.relu(self.fc11(x))
        x = F.relu(self.fc12(x))
        x = F.relu(self.fc13(x))
        x = F.relu(self.fc14(x))
        x = F.relu(self.fc15(x))
        x = F.relu(self.fc16(x))
        x = F.relu(self.fc17(x))
        x = F.relu(self.fc18(x))
        x = F.relu(self.fc19(x))
        x = F.relu(self.fc20(x))
        x = F.relu(self.fc21(x))
        x = F.relu(self.fc22(x))
        return x
    
# Training the model   
def train(model, train_loader, optimizer, criterion, epoch):
    model.train()
    for batch_idx, data in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data[:, :3])
        loss = criterion(output, data[:, 3:])
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
    return loss.item()

# Testing the model
def test(model, test_loader, criterion):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            output = model(data[:, :3])
            test_loss += criterion(output, data[:, 3:]).item() # sum up batch loss
    test_loss /= len(test_loader.dataset)
    return test_loss

# Main
if __name__ == '__main__':
    # Loading the data
    train_dataset = uniCompDataset('uniCompTrain.csv')
    test_dataset = uniCompDataset('uniCompTest.csv')
    train_loader = DataLoader(train_dataset, batch_size=1000, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)
    
    # Creating the model
    model = uniCompNet()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Training the model
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    start = time.time()
    for epoch in range(1, 1000):
        train(model, train_loader, optimizer, criterion, epoch)
        test_loss = test(model, test_loader, criterion)
        print('Test set: Average loss: {:.4f}'.format(test_loss))
        