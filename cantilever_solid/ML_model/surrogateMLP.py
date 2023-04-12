"""
__author__ = "Leopoldo Agorio and Mauricio Vanzulli"
__email__ = "lagorio@fing.edu.uy  mvanzulli@fing.edy.uy"
__status__ = "Development"
__date__ = "04/23"
"""

"""Loading the csv with torch's data loader and using it for batch training
remember the csv structure was:
```
 echo "$Lx, $Ly, $Lz, $E1, $nu1, $E2, $nu2, $p, $Ux, $Uy, $Uz" >> "$filename"
```
 where Lx, Ly, Lz are the block's length, E1, E2, nu1, nu2 and nu are material parameter,  p is the input pressure
 and Ux, Uy, Uz are the output compression """

# Import libraries 
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch.nn as nn
import subprocess

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
        self.loss_val_norm = [] # With norm 2 relative loss
        self.loss_train_norm = [] # With norm 2 relative loss

        self.index_features = [0,3,5,7] # features input is in data frame [:, 0,3,5,7] Lx,E1,E2,p
        self.index_labels = [-3,-2,-1] # label output in data frame [:, -3,-2,-1] Ux,Uy,Uz

        self.device = torch.device('cpu')

    def forward(self, x):
        return self.layers(x)

    def train(self, train_loader, val_loader, optimizer=None, criterion=None, num_epochs=2000, verbose=False):
        """
        Trains a neural network model.

        Args:
            train_loader (DataLoader): The data used for training.
            val_loader (DataLoader): The validation data.
            optimizer (Optimizer, optional): The optimizer used for training. If None, defaults to self.optimizer.
            criterion (Loss function, optional): The loss function used for training. If None, defaults to self.criterion.
            num_epochs (int, optional): The number of epochs for training. Defaults to 2000.
            analytic (bool, optional): A flag to indicate whether to compute an additional analytic loss. Defaults to False.
            verbose (bool, optional): A flag to indicate whether to print updates during training. Defaults to False.

        Returns:
            Tuple: Computed train, validation and analytic loss (if `analytic=True`) as arrays.
        """
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
            val_loss_norm = self.evaluate(val_loader)
            train_loss_norm = self.evaluate(train_loader)
            tqdm.write(f"Epoch: {epoch}, Train Loss: {train_loss}, Train Loss norm: {train_loss_norm}, Val Loss Norm: {val_loss_norm}") if verbose else None

            self.loss.append(train_loss)
            self.loss_train_norm.append(train_loss_norm)
            self.loss_val_norm.append(val_loss_norm)

    """Computes the RMSE on a generic dataset """    
    def evaluate(self, loader):
        val_loss = 0.0
        with torch.no_grad():
            for data in loader:
                data = data.to(self.device)
                predicted = self(data[:, self.index_features])
                ground_truths = data[:, self.index_labels] # ux,uy,uz
                
                norm_dif = torch.norm((predicted -  ground_truths), dim=1, keepdim=True)**2
                norms_ground_truth = torch.norm(ground_truths)**2
                
                val_loss += torch.sum(norm_dif/norms_ground_truth).item() /len(predicted)
        val_loss /= len(loader)
        return val_loss
    
# Main
if __name__ == '__main__':
    # Loading the data
    dataset = cantileverComposedDataset('./cantilever_solid/data/data.csv')

    # Set a seed for reproducibility
    torch.manual_seed(0)

    # Splitting the data into training and validation sets
    train_size = 1000
    val_size = 1000
    test_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])

    # Creating data loaders
    batch_size = 1000
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    # Defining the model
    mlp = MLP()
    # Training the model
    mlp.train(train_loader =train_loader, val_loader=val_loader, num_epochs=150) 

    # Evaluate test loss
    test_loss_RMSE = mlp.evaluate(test_loader) 

    # Print final losses
    print("Final training loss MSE %.6f  " % (mlp.loss[-1]))
    print("Final training loss RMSE %.6f  " % (mlp.loss_train_norm[-1]))
    print("Final validation loss RMSE %.6f  " % (mlp.loss_val_norm[-1]))
    print("Final test loss RMSE %.6f  " % (test_loss_RMSE))

    
    # Compute the extrapolation error
    # This is the most flexible case that ONSAS.m can solve with the parameters
    # defined in the input
    Lx_extra, E1_extra, E2_extra, p_extra = (1.5, 1.2, 1.2, 0.16)

    # Set input values as strings
    Lx = str(Lx_extra)
    E1 = str(E1_extra)
    E2 = str(E2_extra)
    p = str(p_extra)
    nu1 = "0.3"
    nu2 = "0.3"
    Ly = "1.0"
    Lz = "0.5"

    # Run the Octave script with the specified input values
    # Move to the folder cantilever solid
    cmd = f"cd ./cantilever_solid/FEM_model && LC_ALL=C octave -q cantilever_solid.m {Lx} {E1} {E2} {p} {nu1} {nu2} {Ly} {Lz} > cliOutput.txt"
    subprocess.run(cmd, shell=True)

    # Get the ground truths output values from ONSAS for Ux, Uy, and Uz
    with open("./cantilever_solid/FEM_model/output.txt") as file:
        lines = file.readlines()
        ux_extra_gt = float(lines[0].strip())
        uy_extra_gt = float(lines[1].strip())
        uz_extra_gt = float(lines[2].strip())

    # Ground truth
    # ux_extra_gt, uy_extra_gt, uz_extra_gt = (Lx_extra, 1.0, 1.0, E_extra, 0.3, p_extra)    
    u_pred = mlp(torch.tensor([[Lx_extra, E1_extra, E2_extra, p_extra]], dtype=torch.float)).detach().numpy()
    dif = np.array((u_pred[0][0]-ux_extra_gt, u_pred[0][1]-uy_extra_gt, u_pred[0][2]-uz_extra_gt))
    # Compute errors    
    MSE_extra = np.linalg.norm(dif)**2
    RMSE_extra = MSE_extra / np.linalg.norm(np.array((ux_extra_gt, uy_extra_gt, uz_extra_gt)))**2
    
    # Plot in screen the error for Lx_extra E_extra p_extra
    print(
        "RMSE extrapolation error for (Lx, E1, E2, p) = (%.2f, %.2f, %.2f, %.2f): is %.6f" % 
        (Lx_extra, E1_extra, E2_extra, p_extra, RMSE_extra)
    )
    
    ## Plotting the loss
    # Plot params
    color_training = 'tab:blue'
    linestyle_training = '--'
    color_validation = 'tab:orange'

    # Make a subplot with MSE loss and RMSE loss
    fig, (ax1, ax2) = plt.subplots(1, 2)
    plt.style.use("seaborn-v0_8")

    # Plot the mlp.loss in the first subplot
    ax1.semilogy(mlp.loss, label='training', marker=None, color=color_training, linestyle=linestyle_training)
    ax1.legend(loc="upper right")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("MSE")
    ax1.grid(True)
    
    # Plot the mlp norm loss in the second subplot
    ax2.semilogy(mlp.loss_val_norm, label='validation', marker=None, color=color_validation)
    ax2.semilogy(mlp.loss_train_norm, label='training', marker=None, color=color_training, linestyle=linestyle_training)
    ax2.legend(loc="upper right")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("RMSE")
    ax2.grid(True)

    #save the image
    plt.show()
    plt.savefig('./cantilever_solid/lossCantilever.png')

    pass
