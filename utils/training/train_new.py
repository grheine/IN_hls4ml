import torch
import torchmetrics
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR

import copy
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from utils.training.pytorchtools import EarlyStopping
from utils.plotting.plot import watermark



class train_model:

    def __init__(self, train_loader, val_loader, model, optimizer, scheduler, batch_size=1000, epochs=30, patience=5, name='trained_IN', device='cpu'):
        
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        self.name = name
        self.trainedmodel, self.loss, self.val_loss, self.acc, self.val_acc = self.__train_model()
     
        
        
    def __train_model(self):
        print("The training starts now, please be patient")
        # to track the training/ validation loss and last accuracy as the model trains
        train_losses = []
        val_losses = []
        train_accuracy = torchmetrics.Accuracy()
        val_accuracy = torchmetrics.Accuracy()

        # to track the average training/ validation loss and last accuracy per epoch as the model trains
        avg_train_losses = []
        avg_val_losses = [] 
        train_accs = []
        val_accs = [] 

        # initialize the early_stopping object
        early_stopping = EarlyStopping(patience=self.patience, verbose=True)

        for epoch in range(1, self.epochs + 1):
            ###################
            # train the model #
            ###################
            self.model.train() # prep model for training
            for batch, data in enumerate(tqdm(self.train_loader), 1):
                data = data.to(self.device)   
                if (len(data.x)==0): continue   
                target = data.y
                self.optimizer.zero_grad() # clear the gradients of all optimized variables
                output = self.model(data).squeeze(1)
                loss = F.binary_cross_entropy(output,target)
                loss.backward()
                self.optimizer.step()
                # record batch training loss and accuracy
                train_losses.append(loss.item())          
                batch_acc = train_accuracy(output, target.int())

            ######################    
            # validate the model #
            ######################
            self.model.eval() # prep model for evaluation
            for batch, data in enumerate(tqdm(self.val_loader), 1):
                data = data.to(self.device)   
                if (len(data.x)==0): continue   
                target = data.y
                output = self.model(data).squeeze(1)
                loss = F.binary_cross_entropy(output,target)
                # record batch validation loss and accuracy
                val_losses.append(loss.item())
                batch_val_acc = val_accuracy(output, target.int())

            # print training/validation statistics 
            # calculate average loss and accuracy over an epoch
            train_loss = np.average(train_losses)
            val_loss = np.average(val_losses)
            avg_train_losses.append(train_loss)
            avg_val_losses.append(val_loss)

            train_acc = train_accuracy.compute()
            val_acc = val_accuracy.compute()
            train_accs.append(train_acc.item())
            val_accs.append(val_acc.item())

            epoch_len = len(str(self.epochs))

            print_msg = (f'[{epoch:>{epoch_len}}/{self.epochs:>{epoch_len}}] ' +
                         f'train_loss: {train_loss:.5f} ' +
                         f'valid_loss: {val_loss:.5f} ' +
                         f'train_acc: {train_acc:.5f} ' +
                         f'valid_acc: {val_acc:.5f} ')

            print(print_msg)   

            # clear lists to track next epoch
            train_losses = []
            val_losses = []
            train_accuracy.reset()
            val_accuracy.reset()

            # early_stopping needs the validation loss to check if it has decreased, 
            # and if it has, it will make a checkpoint of the current model
            early_stopping(val_loss, self.model)

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
        
        # Decay Learning Rate
        self.scheduler.step()

        return  trained_model, avg_train_losses, avg_val_losses, train_accs, val_accs
    
    def plot_loss(self):
        plt.style.use("kit")
        minloss = min(self.val_loss)
        plt.plot(range(1,len(self.loss)+1),self.loss, label='Training Loss', marker='None')
        plt.plot(range(1,len(self.val_loss)+1),self.val_loss,label='Validation Loss', marker='None')
        plt.plot([], [], ' ', label=f'best = {minloss:.4f}')

        # find position of lowest validation loss
        minposs = self.val_loss.index(min(self.val_loss))+1 
        if minposs != self.epochs:
            plt.axvline(minposs, linestyle=':', color='black',label='Early Stopping')

        watermark(py=0.9, fontsize=18, shift=0.16)
        plt.legend(loc='upper right', frameon = True, framealpha = 1, facecolor = 'white', edgecolor = 'white')
        plt.xlabel('epochs')
        plt.ylabel('loss')
        plt.savefig('img/loss_plot.png', bbox_inches='tight')
        plt.show()
        
    def plot_acc(self):
        plt.style.use("kit")
        maxacc = max(self.val_acc)
        plt.plot(range(1,len(self.acc)+1),self.acc, label='Training Accuracy', marker='None')
        plt.plot(range(1,len(self.val_acc)+1),self.val_acc,label='Validation Accuracy', marker='None')
        plt.plot([], [], ' ', label=f'best = {maxacc:.4f}')
        
        if minposs != self.epochs:
            plt.axvline(minposs, linestyle=':', color='black',label='Early Stopping')

        plt.xlabel('epochs')
        plt.ylabel('accuracy')
        watermark(py=0.9, fontsize=18, shift=0.16)
        plt.legend(loc='lower right', frameon = True, framealpha = 1, facecolor = 'white', edgecolor = 'white')
        plt.savefig('img/acc_plot.png', bbox_inches='tight')
        plt.show()

 
    def show(self):
        #plot loss and accuracy
        self.plot_loss()
        self.plot_acc()