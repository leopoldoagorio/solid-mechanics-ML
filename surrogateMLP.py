"""
__author__ = "Leopoldo Agorio and Mauricio Vanzulli"
__email__ = "lagorio@fing.edu.uy  mvanzulli@fing.edy.uy"
__status__ = "Development"
__date__ = "02/23"
"""

"""Loading the csv with torch's data loader and using it for batch training
remember the csv structure was: echo "$Lx,$Ly,$Lz, $E, $nu, $p,$Ux,$Uy,$Uz" >> "$filename"
where Lx, Ly, Lz are the block's length, E and nu are material parameter,  p is the input pressure
 and Ux, Uy, Uz are the output compression """

 
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
import pandas as pd
from analytic_solution import compute_analytic_solution
import numpy as np
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
        self.activation = torch.nn.ReLU()

        # create an empty sequential model
        self.Net = torch.nn.Sequential()

        # Add first layer 
        self.n_features = 3
        self.n_labels = 3
        self.Net.add_module("input_layer", torch.nn.Linear(self.n_features, self.n_neurons_per_layer))
        self.Net.add_module("hidden_activation_num_{}".format(0), self.activation)

        # Add hidden layers
        for n_hidden_layer in range(self.depth):
            self.Net.add_module(
                "hidden_layer_num_{}".format(n_hidden_layer + 1),
                torch.nn.Linear(self.n_neurons_per_layer, self.n_neurons_per_layer)
            )
            self.Net.add_module("hidden_activation_num_{}".format(n_hidden_layer + 1), self.activation)

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
    
    def forward(self, x):
        return self.Net(x)

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

# Testing the model
def test_analytic(model, criterion):
    Ly = 1
    Lz = 1
    nu = 0.3
    
    test_loss = 0
    correct = 0
    
    # create test data
    Lx_range = np.linspace(0.8, 3.2, 33)
    Em_range = np.linspace(0.5, 4.5, 33)
    pressure_range = -np.linspace(0.05, 3.2, 33)
    test_data = []
    test_labels = []
    for Lx in Lx_range:
        for Em in Em_range:
            for pressure in pressure_range:
                test_data.append([Lx, Em, pressure])
                ux, uy, uz = compute_analytic_solution(Lx, Ly, Lz, Em, nu, pressure)
                test_labels.append([ux, uy, uz])
    test_data = torch.tensor(test_data, dtype=torch.float)
    test_labels = torch.tensor(test_labels, dtype=torch.float)
    test_dataset = TensorDataset(test_data, test_labels)
    test_loader = DataLoader(test_dataset, batch_size=100, shuffle=True)
    
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            test_loss += criterion(output, target).item()
    test_loss /= len(test_loader.dataset)
    print('Test set: Average loss: {:.4f}'.format(test_loss))
    
    return test_loss


# Main
if __name__ == '__main__':
    # Loading the data
    train_dataset = uniCompDataset('data.csv')
    train_loader = DataLoader(train_dataset, batch_size=1000, shuffle=True)
    test_dataset = uniCompDataset('data.csv')
    test_loader = DataLoader(test_dataset, batch_size=100, shuffle=True)

    # Defining the model
    mlp = MLP()
    mlp.train(train_loader, num_epochs=3) 
    # Testing the model
    # mlp.Net with a numpy array of size 3
    Lx = 1
    Ly = 1
    Lz = 1
    Em = 1
    nu = 0.3
    pressure = 1

    mlp.Net(torch.tensor([Lx, Em, pressure], dtype=torch.float))
    #compute analytic solution
    ux, uy, uz = compute_analytic_solution(Lx,Ly,Lz, Em, nu, pressure)


    # print and compare the results
    print("Analytic solution: ", compute_analytic_solution(Lx,Ly,Lz, Em, nu, pressure))
    print("MLP solution: ", mlp.Net(torch.tensor([Lx, Em, pressure], dtype=torch.float)))
    # error
    # [ux, uy, uz] to tensor
    print("Error: ", torch.norm(mlp.Net(torch.tensor([Lx, Em, pressure], dtype=torch.float)) - torch.tensor([ux, uy, uz], dtype=torch.float)))
    
    # test the model
    result = test_analytic(mlp, torch.nn.MSELoss())
    print("Test result: ", result)

    # train some more
    morep = 10
    mlp.train(train_loader, num_epochs=3)
    result = test_analytic(mlp, torch.nn.MSELoss())
    print("After {} epochs, test result: ".format(morep), result)


    pass