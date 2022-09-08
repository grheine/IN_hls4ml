import os
import torch
import torchmetrics
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

from time import time
from tqdm import tqdm
import copy

import numpy as np

from utils.training.pytorchtools import EarlyStopping

class train_model:
    
    def __init__(self, train_loader, val_loader, model, optimizer, scheduler, device='cpu', epochs=30, patience=5, run='full_graph', name='IN_trained'):
        
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.epochs = epochs
        self.patience = patience
        self.run = run
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
        val_losses = []
        with torch.no_grad():                # disableing gradient calculation for inference 
            for data in self.val_loader:
                data = data.to(self.device)
                if (len(data.x)==0): continue 
                target = data.y
                output = self.model(data).squeeze(1)
                val_loss = F.binary_cross_entropy(output,target)
                val_acc.update(output, target.int())
                val_losses.append(val_loss.item())
                
        print('\n Validation set: Average loss: {:.4f}\n, Accuracy: {:.4f}\n'
              .format(np.mean(val_losses), val_acc.compute()))
        
        return np.nanmean(val_losses), val_acc.compute()

    def train_model(self):
        # initialize the early_stopping object
        os.makedirs(f'models/{self.run}', exist_ok=True)
        os.makedirs(f'train_output/{self.run}', exist_ok=True)

        early_stopping = EarlyStopping(patience=self.patience, verbose=True, path=f'models/checkpoint_{self.run}.pt')

        losses, accs = [], []
        val_losses, val_accs = [], []       
        output = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

        for epoch in tqdm(range(1, self.epochs + 1)):
            train_loss, train_acc = self.train(epoch)
            val_loss, val_acc = self.validate(threshold=0.5)
            
            self.scheduler.step()

            # early_stopping needs the validation loss to check if it has decreased, 
            # and if it has, it will make a checkpoint of the current model
            early_stopping(val_loss, self.model)
            if early_stopping.early_stop:
                print("Early stopping")
                break
                # load the last checkpoint with the best model
         
            output['train_loss'].append(train_loss)
            output['train_acc'].append(train_acc)
            output['val_loss'].append(val_loss)
            output['val_acc'].append(val_acc)

        nevents = len(self.train_loader)
        np.save(f'train_output/{self.run}/{self.name}', output)
                
        #save best model
        self.model.load_state_dict(torch.load(f'models/checkpoint_{self.run}.pt'))
        trained_model = copy.deepcopy(self.model)

        torch.save(self.model.state_dict(), f"models/{self.run}/{self.name}_state_dict.pt")
        torch.save(self.model, f'models/{self.run}/{self.name}.pt')
        
        return  self.model, output