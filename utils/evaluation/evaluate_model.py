import torch
import numpy as np
from sklearn.metrics import roc_curve, auc, ConfusionMatrixDisplay, confusion_matrix

import matplotlib.pyplot as plt
from utils.plotting.plot import watermark



class evaluate_model:
    
    def __init__(self, model, test_data, output=None, cut=0.5, ncuts=100):
       
        self.model = model
        self.train_loss = output['train_loss']
        self.val_loss = output['val_loss']
        self.train_acc = output['train_acc']
        self.val_acc = output['val_acc'] 
        self.cut = cut
        self.minposs = -1
        self.test_data = test_data
        self.target, self.output,  self.MCtrue_output, self.MCfalse_output = self.__GNNoutput__()
        self.ncuts = ncuts
        
        
    def __GNNoutput__(self):
        MCtrue, MCfalse, targets, outputs = [], [], [], []
        for data in self.test_data:
            data = data
            target = data.y
            output = self.model(data).squeeze(1).detach().numpy() 
            MCtrue_output = output[target==True]
            MCfalse_output = output[target!=True]
            MCtrue+=list(MCtrue_output)
            MCfalse+=list(MCfalse_output)
            targets+= list(target.detach().numpy())
            outputs+= list(output)
        return targets, np.array(outputs), MCtrue, MCfalse

        
    def plot_loss(self, early=True):
        plt.style.use("kit")
        n_epochs = len(self.train_loss)
        minloss = min(self.val_loss)
        plt.plot(range(1,len(self.train_loss)+1),self.train_loss, label='Training Loss', marker='None')
        plt.plot(range(1,len(self.val_loss)+1),self.val_loss,label='Validation Loss', marker='None')
        plt.plot([], [], ' ', label=f'best = {minloss:.4f}')

        # find position of lowest validation loss
        self.minposs = self.val_loss.index(min(self.val_loss))+1 
        if self.minposs != n_epochs and early:
            plt.axvline(self.minposs, ymax=0.85, linestyle=':', color='black',label='Early Stopping')

        watermark(py=0.9, fontsize=18, shift=0.16)
        plt.legend(loc='upper right', frameon = True, framealpha = 1, facecolor = 'white', edgecolor = 'white')
        plt.xlabel('epochs')
        plt.ylabel('loss')
        plt.savefig('img/loss_plot.png', bbox_inches='tight')
        plt.show()
        
    def plot_acc(self, early=True):
        plt.style.use("kit")
        n_epochs = len(self.train_loss)
        maxacc = max(self.val_acc)
        plt.plot(range(1,len(self.train_acc)+1),self.train_acc, label='Training Accuracy', marker='None')
        plt.plot(range(1,len(self.val_acc)+1),self.val_acc,label='Validation Accuracy', marker='None')
        plt.plot([], [], ' ', label=f'best = {maxacc:.4f}')
        
        # find position of lowest validation loss
        if self.minposs != n_epochs and early:
            plt.axvline(self.minposs, ymax=0.85, linestyle=':', color='black',label='Early Stopping')

        plt.xlabel('epochs')
        plt.ylabel('accuracy')
        watermark(py=0.9, fontsize=18, shift=0.16)
        plt.legend(loc='lower right', frameon = True, framealpha = 1, facecolor = 'white', edgecolor = 'white')
        plt.savefig('img/acc_plot.png', bbox_inches='tight')
        plt.show()
        
    def plot_GNNoutput(self):
        plt.style.use("kit_hist")
        plt.hist([self.MCtrue_output, self.MCfalse_output], bins=30, histtype='stepfilled', facecolor='white',stacked=True, label=['MC true edge', 'MC false edge'])
        plt.yscale('log')
#         plt.ylim(top=10e5)
        watermark(py=0.9, fontsize=18, shift=0.16)
        plt.xlabel('GNN output')
        plt.ylabel('counts')
        plt.legend()
        plt.savefig('img/GNNoutput.png', bbox_inches='tight')
        plt.show()
        
    def plot_roc(self):
        plt.style.use("kit")
        fpr, tpr, thresholds = roc_curve(self.target, self.output)
        auc_val=auc(fpr, tpr)
        
        plt.plot(fpr, tpr, label='AUC = {1:.4f}'.format('$p$',auc_val), marker='None')
        watermark(py=0.9, fontsize=18, shift=0.16)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend(loc='center right', bbox_to_anchor=(1, 0.91))
        plt.savefig('img/roc_plot.png', bbox_inches='tight')
        plt.show()
        
    def plot_confm(self):
        plt.style.use("kit")
        output_conf = (self.output>self.cut)*1
        confm = confusion_matrix(self.target,output_conf, normalize='all')
        dispConf = ConfusionMatrixDisplay(confusion_matrix=confm, display_labels=['True', 'False'])
        dispConf.plot()
        plt.xlabel(f'Predicted label (cut @ {self.cut:.2f})')
        plt.ylabel('MC True label')
        plt.yticks(rotation = 90)
        plt.savefig('img/confusion_matrix.png', bbox_inches='tight')
        plt.show()
        
    def plot_TPR_PPV(self):
        cuts = np.linspace(0.01, 0.99, self.ncuts)
        TPR = []
        PPV = []

        for c in cuts:
            MCtrue = np.array(self.MCtrue_output, dtype=np.float32)
            TP = len(MCtrue[MCtrue>c])
            P = np.sum(self.target)
            npassed = len(self.output[self.output>c])
            TPR.append(TP/P)
            if npassed == 0:
                PPV.append(0)
            else:
                PPV.append(TP/npassed)
            
        plt.style.use("kit")        
        plt.plot(cuts, TPR, label='purity', marker='None')
        plt.plot(cuts, PPV, label='efficiency', marker='None')
        watermark(py=0.9, fontsize=18, shift=0.16)
        plt.xlabel('cut')
        plt.ylabel('GNN output')
#         plt.ylim(0.96,1.015)
        plt.legend(loc='center right', bbox_to_anchor=(1, 0.91))
        plt.savefig('img/purity_efficiency.pdf', bbox_inches='tight')
        plt.show()      
        
        
    def plot_metrics(self):
        self.plot_GNNoutput()
        self.plot_roc()
        self.plot_confm()
        self.plot_TPR_PPV()