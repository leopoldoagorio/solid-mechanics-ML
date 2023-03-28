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
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch.nn as nn
import pyDOE

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
class MLP(nn.Module):
    def __init__(self, input_dim=3, output_dim=3, hidden_layers=[20, 10]):
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
        self.loss_train_norm = []
        self.loss_val_norm = []
        self.loss_analytic_norm = []

        self.device = torch.device('cpu')

    def forward(self, x):
        return self.layers(x)

    def train(self, train_loader, val_loader, optimizer=None, criterion=None, num_epochs=2000, analytic=False,verbose=False):
        if optimizer is None:
            optimizer = self.optimizer
        if criterion is None:
            criterion = self.criterion
        if(analytic):
            test_data, test_labels = generate_test_dataset_lhs(Ly=1, Lz=1, nu=0.3)
            test_dataset = TensorDataset(test_data, test_labels)
            test_loader = DataLoader(test_dataset, batch_size=100, shuffle=True)

        for epoch in tqdm(range(num_epochs)):
            train_loss = 0.0
            for batch_idx, data in enumerate(train_loader):
                data = data.to(self.device)
                optimizer.zero_grad()
                predicted = self(data[:, [0,3,5]])# input is in data[:, 0,3,5] Lx,E,p
                ground_truths = data[:, -3:] # ux,uy,uz
                loss = criterion(predicted, ground_truths)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            train_loss /= len(train_loader)
            val_loss_norm = self.val(val_loader, criterion)
            train_loss_norm = self.val(train_loader, criterion)
            if(analytic==False and verbose):
                tqdm.write(f"Epoch: {epoch}, Train Loss: {train_loss}, Val Loss: {val_loss}")

            self.loss.append(train_loss)
            self.loss_train_norm.append(train_loss_norm)
            self.loss_val_norm.append(val_loss_norm)

            if(analytic):
                self.loss_analytic_norm.append(test_analytic(self, criterion,test_loader))
                if(verbose):
                    tqdm.write(f"Epoch: {epoch}, Train Loss: {train_loss}, Val Loss norm: {val_loss_norm}, Analytic Loss: {self.loss_analytic_norm[-1]}")

    def val(self, val_loader, criterion):
        #self.eval()
        val_loss = 0.0
        with torch.no_grad():
            for data in val_loader:
                data = data.to(self.device)
                predicted = self(data[:, [0,3,5]])# input is in data[:, 0,3,5] Lx,E,p
                ground_truths = data[:, -3:] # ux,uy,uz

                norm_dif = torch.norm((predicted -  ground_truths), dim=1, keepdim=True)**2
                norms_ground_truth = torch.norm(ground_truths)**2
                
                val_loss += torch.sum(norm_dif/norms_ground_truth).item() /len(predicted)
        val_loss /= len(val_loader)
        return val_loss

def generate_test_dataset_lhs(Ly, Lz, nu, samples = 200):
    Lx_range = [1, 3]
    Em_range = [1.0, 4.0]
    pressure_range = [0.1, 3.0]
    design = pyDOE.lhs(3, samples=samples, criterion='maximin')
    test_data = []
    test_labels = []
    for i in range(samples):
        Lx = design[i][0]*(Lx_range[1]-Lx_range[0])+Lx_range[0]
        Em = design[i][1]*(Em_range[1]-Em_range[0])+Em_range[0]
        pressure = design[i][2]*(pressure_range[1]-pressure_range[0])+pressure_range[0]
        test_data.append([Lx, Em, pressure])
        ux, uy, uz = compute_analytic_solution(Lx, Ly, Lz, Em, nu, pressure)
        test_labels.append([ux, uy, uz])
    test_data = torch.tensor(test_data, dtype=torch.float)
    test_labels = torch.tensor(test_labels, dtype=torch.float)
    return test_data, test_labels

# Testing the model
def test_analytic(model, criterion, test_loader):
    test_loss = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
        
            norm_dif = torch.norm((output -  target), dim=1, keepdim=True)**2
            norms_ground_truth = torch.norm(target)**2
        
            test_loss += torch.sum(norm_dif/norms_ground_truth).item() /len(target)

    test_loss /= len(test_loader)    
    return test_loss


# Main
if __name__ == '__main__':
    # Loading the data
    dataset = uniCompDataset('./uniaxial_compression/data/data.csv')#uniCompDataset('./../data/data.csv')

    # Splitting the data into training and validation sets
    frac_train = 0.5
    train_size = int(frac_train * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    # Creating data loaders
    batch_size = 1000
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Defining the model
    mlp = MLP()
    mlp.train(train_loader, val_loader=val_loader, num_epochs=100, analytic = True) 

    ## Plotting the loss
    # Make a subplot with mlp.loss and mlp.loss_val_norm
    fig, (ax1, ax2) = plt.subplots(1, 2)
    plt.style.use("seaborn-v0_8")

    # Plot the mlp.loss in the first subplot
    ax1.semilogy(mlp.loss, label='training', marker=None)
    ax1.set_title("Training Loss MSE")
    ax1.legend(loc="upper right")
    ax1.set_xlabel("Epoch")
    ax1.grid(True)
    
    # Plot the mlp norm loss in the second subplot
    ax2.semilogy(mlp.loss_val_norm, label='validation', marker=None)
    ax2.semilogy(mlp.loss_train_norm, label='training', marker=None)
    ax2.semilogy(mlp.loss_analytic_norm, label='analytic', marker=None)
    ax2.legend(loc="upper right")
    ax2.set_title("train, validation and analytic relative error")
    ax2.set_xlabel("Epoch")
    ax2.grid(True)


    #save the image
    plt.show()
    plt.savefig('./lossUniaxial.png')


    # Compute the error versus the NN and ONSAS solution for an outlier (most flexible case).
    pred = mlp(torch.tensor([[2.9, 0.5, 2.9]], dtype=torch.float)).detach().numpy()
    ux, uy, uz = compute_analytic_solution(2.9, 1.0, 1.0, 0.5, 0.3, 2.9)
    print('NN solution: ', pred)
    print('Analytic solution: ', ux, uy, uz)
    error = np.sqrt((pred[0][0]-ux)**2+(pred[0][1]-uy)**2+(pred[0][2]-uz)**2)/np.sqrt(ux**2+uy**2+uz**2)
    print('Error: ', error)
    #relative error
    


    pass