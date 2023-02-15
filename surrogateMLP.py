
#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

__author__ = "Leopoldo Agorio and Mauricio Vanzulli"
__email__ = "lagorio@fing.edu.uy  mvanzulli@fing.edy.uy"
__status__ = "Development"
__date__ = "02/23"

"""Loading the csv with torch's data loader and using it for batch training
remember the csv structure was: echo "$Lx,$Ly,$Lz, $E, $nu, $p,$Ux,$Uy,$Uz" >> "$filename"
where Lx, Ly, Lz are the block's length, E and nu are material parameter,  p is the input pressure
 and Ux, Uy, Uz are the output compression """

 
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
# Loading the data
#echo "$Lx,$Ly,$Lz,$E,$nu,$p,$Ux,$Uy,$Uz" >> "$filename"
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
class MLP(torch.nn.Module):
    def __init__(self, depth = 5, n_neurons_per_layer=8):
        super(MLP, self).__init__()
        self.depth = depth
        self.loss = []
        self.n_neurons_per_layer = n_neurons_per_layer
        
        # create an empty sequential model
        self.Net = torch.nn.Sequential()

        # Add first layer 
        self.n_features = 3
        self.n_labels = 3
        self.Net.add_module("input_layer", torch.nn.Linear(self.n_features, self.n_neurons_per_layer))
        
        # Add hidden layers
        for n_hidden_layer in range(self.depth):
            self.Net.add_module(
                "hidden_layer_num_{}".format(n_hidden_layer + 1),
                torch.nn.Linear(self.n_neurons_per_layer, self.n_neurons_per_layer)
            )
            self.Net.add_module("hidden_activation_num_{}".format(n_hidden_layer + 1 ), torch.nn.ReLU())

        # Add output layer
        self.Net.add_module("output_layer", torch.nn.Linear(self.n_neurons_per_layer, self.n_labels))
        
        # Initialize weights iterating over all the modules
        for m in self.Net.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                torch.nn.init.zeros_(m.bias)

        # Define training loop 
        self.optimizer = torch.optim.Adam(self.Net.parameters(), lr=0.001)
        self.criterion = torch.nn.MSELoss()

        # Define device
        # self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.device = "cpu"

    def train(self, train_loader, optimizer=None, criterion=None, num_epochs=2000):
        if optimizer is None:
            optimizer = self.optimizer
        if criterion is None:
            criterion = self.criterion

        for epoch in range(num_epochs):
            for batch_idx, data in enumerate(train_loader):
                #print(batch_idx) 0....18
                data = data.to(self.device)
                optimizer.zero_grad()
                predicted = self.Net(data[:, :3]) # Lx,E,p
                ground_truths = data[:, :3] # Lx,E,p
                loss = criterion(predicted, ground_truths)
                loss.backward()
                optimizer.step()
                # print statistics when batch_idx is 18 (only 18 batches in the dataset)
            print(f"Epoch: {epoch}, Loss: {loss.item()}")  
            self.loss.append(loss.item())


# # Testing the model
# def test(model, test_loader, criterion):
#     model.eval()
#     test_loss = 0
#     correct = 0
#     with torch.no_grad():
#         for data in test_loader:
#             data = data.to(device)
#             output = model(data[:, :3])
#             test_loss += criterion(output, data[:, 3:]).item() # sum up batch loss
#     test_loss /= len(test_loader.dataset)
#     return test_loss

# Main
if __name__ == '__main__':
    # Loading the data
    train_dataset = uniCompDataset('data.csv')
    train_loader = DataLoader(train_dataset, batch_size=1000, shuffle=True)

    # Defining the model
    mlp = MLP()
    mlp.train(train_loader) 
    pass