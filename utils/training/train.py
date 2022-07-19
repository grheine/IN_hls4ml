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
    
    def __init__(self, train_loader, val_loader, test_loader, model, optimizer, scheduler, device='cpu', epochs=30, patience=5, name='trained_IN'):
        
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device
        self.epochs = epochs
        self.patience = patience
        self.name = name
        self.trained = self.train_model()

    def train(self, epoch):

        self.model.train()
        losses, accs = [], []
        epoch_t0 = time()
        accuracy = 0

        for batch_idx, data in enumerate(self.train_loader):
            data = data.to(self.device)       
            if (len(data.x)==0): continue        
            target = data.y
            self.optimizer.zero_grad()
            output = self.model(data).squeeze(1)
            loss = F.binary_cross_entropy(output,target)
            loss.backward()
            self.optimizer.step()

            accuracy += torch.sum(((target==1).squeeze() & 
                                       (output>0.5).squeeze()) |
                                      ((target==0).squeeze() & 
                                       (output<0.5).squeeze())).float()/len(target)
            losses.append(loss.item())
        accuracy /= len(self.train_loader.dataset)

        print(f'Train Epoch: {epoch}, epoch time: {time()-epoch_t0:.2f}s')
        print(f'Mean Train Batch Loss: {np.nanmean(losses):.04f}')

        return np.nanmean(losses), accuracy

    def validate(self):
        self.model.eval()

        best_discs = []
        for data in self.val_loader:
            data = data.to(self.device)
            if (len(data.x)==0): continue 
            target = data.y
            output = self.model(data).squeeze(1)

            N_correct = torch.sum((target==1).squeeze() & (output>0.5).squeeze())
            N_correct += torch.sum((target==0).squeeze() & (output<0.5).squeeze())
            N_total = len(target)

            diff, best_disc = 100, 0
            best_tpr, best_tnr = 0, 0
            for disc in np.arange(0.2, 0.8, 0.01):
                true_pos = ((target==1).squeeze() & (output>disc).squeeze())
                true_neg = ((target==0).squeeze() & (output<disc).squeeze())
                false_pos = ((target==0).squeeze() & (output>disc).squeeze())
                false_neg = ((target==1).squeeze() & (output<disc).squeeze())
                N_tp, N_tn = torch.sum(true_pos).item(), torch.sum(true_neg).item()
                N_fp, N_fn = torch.sum(false_pos).item(), torch.sum(false_neg).item()
                true_pos_rate = N_tp/(N_tp + N_fn)
                if N_tn+N_fp>0:
                    true_neg_rate = N_tn/(N_tn + N_fp)
                else: true_neg_rate = 0
                delta = abs(true_pos_rate - true_neg_rate)
                if (delta < diff):
                    diff, best_disc = delta, disc 
            best_discs.append(best_disc)
        print(f'TPR = {true_pos_rate}, TNR = {true_neg_rate}')
        print(f'diff= {diff:.04f}, nbest_disc= {best_disc:.04f}')
        return np.nanmean(best_discs)

    def test(self, disc=0.5):
        self.model.eval()
        test_loss = 0
        accuracy = 0
        losses = []
        MCtrue, MCfalse = [], []
        with torch.no_grad():
            for data in self.test_loader:
                data = data.to(self.device)
                if (len(data.x)==0): continue 
                target = data.y
                output = self.model(data).squeeze(1)

                N_correct = torch.sum((target==1).squeeze() & (output>0.5).squeeze())
                N_correct += torch.sum((target==0).squeeze() & (output<0.5).squeeze())
                N_total = len(target)

                MCtrue.append(output[target==1])
                MCfalse.append(output[target==0])

                accuracy += torch.sum(((target==1).squeeze() & 
                                       (output>disc).squeeze()) |
                                      ((target==0).squeeze() & 
                                       (output<disc).squeeze())).float()/len(target)
                test_loss = F.binary_cross_entropy(output,target)
                losses.append(test_loss.item())

        test_loss /= len(self.test_loader.dataset)
        accuracy /= len(self.test_loader.dataset)
        print('\nTest set: Average loss: {:.4f}\n, Accuracy: {:.4f}\n'
              .format(test_loss, accuracy))
        return np.nanmean(losses), accuracy, MCtrue, MCfalse

    def train_model(self):
        # initialize the early_stopping object
        early_stopping = EarlyStopping(patience=self.patience, verbose=True)

        losses, accs = [], []
        test_losses, test_accs = [], []

        epochs = self.epochs

        for epoch in tqdm(range(1, epochs + 1)):
            train_loss, train_acc = self.train(epoch)
            losses.append(train_loss)
            accs.append(train_acc)
            disc = self.validate()
            print(f'mean best disc {disc:.3f}')
            test_loss, test_acc, MCtrue, MCfalse = self.test(disc=disc)
            test_losses.append(test_loss)
            test_accs.append(test_acc)
            self.scheduler.step()

            # early_stopping needs the validation loss to check if it has decreased, 
            # and if it has, it will make a checkpoint of the current model
            early_stopping(test_loss, self.model)

            if early_stopping.early_stop:
                print("Early stopping")
                break
                # load the last checkpoint with the best model
                
        self.model.load_state_dict(torch.load('models/checkpoint.pt'))
        
        #save model
        trained_model = copy.deepcopy(self.model)
        trained_model.eval()
        torch.save(trained_model.state_dict(), f"models/{self.name}_state_dict.pt")
        torch.save(trained_model, f'models/{self.name}.pt')
        
        return  trained_model, losses, accs, disc, test_losses, test_accs