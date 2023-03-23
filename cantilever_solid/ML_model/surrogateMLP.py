"""
__author__ = "Leopoldo Agorio and Mauricio Vanzulli"
__email__ = "lagorio@fing.edu.uy  mvanzulli@fing.edy.uy"
__status__ = "Development"
__date__ = "03/23"
"""

"""Loading the csv with torch's data loader and using it for batch training
remember the csv structure was:
```
 echo "$Lx, $Ly, $Lz, $E1, $nu1, $E2, $nu2, $p, $Ux, $Uy, $Uz" >> "$filename"
```
 where Lx, Ly, Lz are the block's length, E1, E2, nu1, nu2 and nu are material parameter,  p is the input pressure
 and Ux, Uy, Uz are the output compression """

 
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch.nn as nn
import pyDOE

# Loading the data
#  echo "$Lx, $Ly, $Lz, $E1, $nu1, $E2, $nu2, $p, $Ux, $Uy, $Uz" >> "$filename"
class cantileverComposedDataset(Dataset):
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file, header=None)
        self.root = csv_file
        self.len = self.data.shape[0]
    def __getitem__(self, index):
        return torch.tensor(self.data.iloc[index, :].values, dtype=torch.float)
    def __len__(self):
        return self.len

# Creating the model
class MLP(nn.Module):
    def __init__(self, input_dim=4, output_dim=3, hidden_layers=[20, 10]):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_layers = hidden_layers

        layers = []
        prev_layer_size = input_dim
        for layer_size in hidden_layers:
            layers.append(nn.Linear(prev_layer_size, layer_size))
            layers.append(nn.ReLU())
            prev_layer_size = layer_size
        layers.append(nn.Linear(prev_layer_size, output_dim))
        self.layers = nn.Sequential(*layers)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        self.criterion = nn.MSELoss()

        self.loss = []
        self.loss_val = []

        self.index_features = [0,3,5,7] # features input is in data frame [:, 0,3,5,7] Lx,E1,E2,p
        self.index_labels = [-3,-2,-1] # label output in data frame [:, -3,-2,-1] Ux,Uy,Uz

        # if torch.cuda.is_available():
        #     self.device = torch.device('cuda')
        # else:
        #     self.device = torch.device('cpu')
        self.device = torch.device('cpu')

    def forward(self, x):
        return self.layers(x)

    """ Trains the model """
    def train(self, train_loader, val_loader, optimizer=None, criterion=None, num_epochs=2000, verbose=False):
        if optimizer is None:
            optimizer = self.optimizer
        if criterion is None:
            criterion = self.criterion
        
        for epoch in tqdm(range(num_epochs)):
            train_loss = 0.0
            for _, data in enumerate(train_loader):
                data = data.to(self.device)
                optimizer.zero_grad()
                predicted = self(data[:, self.index_features])
                ground_truths = data[:, self.index_labels] 
                loss = criterion(predicted, ground_truths)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            train_loss /= len(train_loader)
            val_loss = self.evaluate(val_loader, criterion)
            tqdm.write(f"Epoch: {epoch}, Train Loss: {train_loss}, Val Loss: {val_loss}") if verbose else None

            self.loss.append(train_loss)
            self.loss_val.append(val_loss)

    """Computes the loss on a generic dataset """    
    def evaluate(self, loader, criterion):
        val_loss = 0.0
        with torch.no_grad():
            for data in loader:
                data = data.to(self.device)
                predicted = self(data[:, self.index_features])
                ground_truths = data[:, self.index_labels] # ux,uy,uz
                loss = criterion(predicted, ground_truths)
                val_loss += loss.item()
        val_loss /= len(loader)
        return val_loss
    
    """ Compte the loss on the null model  """
    def evaluate_null(self, loader, criterion):
        baseline_loss = 0.0
        with torch.no_grad():
            for data in loader:
                data = data.to(self.device)
                predicted = torch.zeros_like(data[:, self.index_labels])
                ground_truths = data[:, self.index_labels]
                loss = criterion(predicted, ground_truths)
                baseline_loss += loss.item()
            
        baseline_loss /= len(loader)
        return baseline_loss

# Main
if __name__ == '__main__':
    # Loading the data
    dataset = cantileverComposedDataset('./cantilever_solid/data/data.csv')

    # Splitting the data into training and validation sets
    # splitted_samples = 1000 
    # frac_train = .5
    # train_size = int(splitted_samples * frac_train)
    train_size = 250
    val_size = 3000
    test_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])

    # Creating data loaders
    batch_size = 1000
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    # Print a summary of a dataset 

    # Defining the model
    mlp = MLP()
    # Training the model
    mlp.train(train_loader =train_loader, val_loader=val_loader, num_epochs=200) 

    ## Plotting the loss
    plt.style.use("seaborn-v0_8")
    plt.semilogy(mlp.loss, label='Training Loss', marker='o')
    plt.semilogy(mlp.loss_val, label='Validation Loss', marker='s')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss and Validation Loss')
    plt.show()

    #save the image
    plt.savefig('./loss.png')

    # Evaluate test loss
    test_loss = mlp.evaluate(test_loader, mlp.criterion) 
    # Print the final losses 
    print('Train loss: ', mlp.loss[-1])
    print('Validation loss: ', mlp.loss_val[-1])
    print('Test loss: ', test_loss)

    # Compute baseline loss
    baseline_loss = mlp.evaluate_null(test_loader, mlp.criterion)
    print('Baseline loss: ', baseline_loss)

    # Print relative losses
    print('Train Relative loss: ', mlp.loss[-1]/baseline_loss)
    print('Validation Relative loss: ', mlp.loss_val[-1]/baseline_loss)
    print('Test Relative loss: ', test_loss/baseline_loss)

    pass
