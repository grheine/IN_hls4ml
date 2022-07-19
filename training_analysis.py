import os
import pandas as pd
import numpy as np
from time import time
from tqdm import tqdm
import copy

import torch
import torch_geometric
torch.cuda.is_available()
from torch_geometric.loader import DataLoader
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import torchmetrics
import torch.nn as nn
import torch.nn.functional as F
torch.manual_seed(42)

from utils.data.load_graphs import load_graphs
from utils.data.graphdata import GraphDataset
from utils.models.interaction_network import InteractionNetwork
# from utils.training.train_3 import train_model
from utils.training.pytorchtools import EarlyStopping
from utils.evaluation.evaluate_model import evaluate_model

import optuna

class train_model:
    
    def __init__(self, trial, train_loader, val_loader, model, optimizer, scheduler, device='cpu', epochs=30, patience=5, name='trained_IN'):
        
        self.trial
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.epochs = epochs
        self.patience = patience
        self.name = name
        self.trained = self.train_model()

    def train(self, epoch):

        self.model.train()
        losses = []
        epoch_t0 = time()
        acc = torchmetrics.Accuracy()

        for batch_idx, data in enumerate(self.train_loader):
            data = data.to(self.device)       
            if (len(data.x)==0): continue        
            target = data.y
            self.optimizer.zero_grad()
            output = self.model(data).squeeze(1)
            loss = F.binary_cross_entropy(output,target)
            loss.backward()
            self.optimizer.step()
            acc.update(output, target.int())
            losses.append(loss.item())      

        print(f'Train Epoch: {epoch}, epoch time: {time()-epoch_t0:.2f}s')
        print(f'Mean Train Batch Loss: {np.nanmean(losses):.04f}')

        return np.nanmean(losses), acc.compute()

    
    def validate(self, threshold=0.5):
        self.model.eval()
        val_loss = 0
        val_acc = torchmetrics.Accuracy(threshold)
        losses = []
        with torch.no_grad():                # disableing gradient calculation for inference 
            for data in self.val_loader:
                data = data.to(self.device)
                if (len(data.x)==0): continue 
                target = data.y
                output = self.model(data).squeeze(1)
                val_loss = F.binary_cross_entropy(output,target)
                val_acc.update(output, target.int())
                losses.append(val_loss.item())

        val_loss /= len(self.val_loader.dataset)
        print('\nTest set: Average loss: {:.4f}\n, Accuracy: {:.4f}\n'
              .format(val_loss, val_acc.compute()))
        return np.nanmean(losses), val_acc.compute()

    def train_model(self):
        # initialize the early_stopping object
        early_stopping = EarlyStopping(patience=self.patience, verbose=True)

        losses, accs = [], []
        val_losses, val_accs = [], []

        for epoch in tqdm(range(1, self.epochs + 1)):
            for i in range(iterations):
                train_loss, train_acc = self.train(epoch)
                losses.append(train_loss)
                accs.append(train_acc)

            val_loss, val_acc = self.validate(threshold=0.5)
            val_losses.append(val_loss)
            val_accs.append(val_acc)
            
            self.scheduler.step()

            # early_stopping needs the validation loss to check if it has decreased, 
            # and if it has, it will make a checkpoint of the current model
            early_stopping(val_loss, self.model)

            if early_stopping.early_stop:
                print("Early stopping")
                break
                # load the last checkpoint with the best model

            trial.report(val_loss, epoch)

            # Handle pruning based on the intermediate value.
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()
                
        self.model.load_state_dict(torch.load('models/checkpoint.pt'))
        
        #save model
        trained_model = copy.deepcopy(self.model)

        torch.save(self.model.state_dict(), f"models/optimization/{self.name}_state_dict.pt")
        torch.save(self.model, f'models/optimization/{self.name}.pt')
        
        return  self.model, losses, accs, val_losses, val_accs




def get_model(trial):
    hidden_dim = trial.suggest_int('hidden_dim', 4, 32)
    print(hidden_dim)
    model = InteractionNetwork(hidden_size=hidden_dim).to(device)

    return model, hidden_dim



def objective(trial):

    nevents = trial.suggest_int("nevents", 100, 30000)
    graphs = load_graphs('data/graphs', 30000)
    print(nevents)
    lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
    print(lr)
    gamma = 0.7
    step_size = 5

    batch_size = trial.suggest_int('batch_size', 1,128, log=True)
    print(batch_size)

    parts = [0.8, 0.1, 0.1] #sizes of training, validation and testing samples
    load_params = {'batch_size': batch_size, 'shuffle': True}
    parts = np.cumsum((nevents*np.array(parts)).astype(int))
    partition = {'train': graphs[:parts[0]],  
                    'test':  graphs[parts[0]:parts[1]],
                    'val': graphs[parts[1]:parts[2]]}
    train_set = GraphDataset(partition['train'])
    train_loader = DataLoader(train_set, **load_params)
    test_set = GraphDataset(partition['test'])
    test_loader = DataLoader(test_set, **load_params)
    val_set = GraphDataset(partition['val'])
    val_loader = DataLoader(val_set, **load_params)

    model, hidden_dim = get_model(trial).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = StepLR(optimizer, step_size=stepsize, gamma=gamma)

    name = f'IN_{nevents}_{hidden_dim}_{lr}_{batch_size}'

    results = train_model(trial, train_loader, val_loader, test_loader, model, optimizer, scheduler, epochs=epochs, patience=patience, name=name)
    model, losses, accs, val_loss, val_acc = results.trained

    return val_loss



if __name__ == "__main__":

    device = torch.device('cuda:2')
    epochs = 10
    iterations = 10


    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=10, catch=(ValueError,))

    pruned_trials = [t for t in study.trials if t.state ==
                     optuna.trial.TrialState.PRUNED]
    complete_trials = [t for t in study.trials if t.state ==
                       optuna.trial.TrialState.COMPLETE]

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

    f = open(f'optimization_params.txt', 'w')
    for key, value in trial.params.items():
        f.write(key)
        f.write(': ')
        f.write(f'{value}')
        f.write('\n')







