import torch
import numpy as np
from sklearn.metrics import roc_curve, auc, ConfusionMatrixDisplay, confusion_matrix
from tqdm import tqdm

import matplotlib.pyplot as plt
from utils.plotting.plot import watermark, infotext



class evaluate_model:
    
    def __init__(self, model, test_data, pz_min, slope_max, nevents, train_output={}, cut=0.5, ncuts=100, name='', title=''):
        self.name = name
        self.title = title
        self.model = model
        self.infos = infotext(None, pz_min, slope_max, None, nevents)
        self.infos2 = infotext(ntest=len(test_data))
        self.train_output = train_output
        self.cut = cut
        self.minposs = -1
        self.test_data = test_data
        self.target, self.output,  self.MCtrue_output, self.MCfalse_output = self.__GNNoutput__()
        self.ncuts = ncuts

        
    def __GNNoutput__(self):
        MCtrue, MCfalse, targets, outputs = [], [], [], []
        for data in self.test_data:
            target = data.y
            try:
                output = self.model(data).squeeze(1).detach().numpy()
            except:
                print('error')
                continue

            MCtrue_output = output[target==True]
            MCfalse_output = output[target!=True]
            MCtrue+=list(MCtrue_output)
            MCfalse+=list(MCfalse_output)
            targets+= list(target.detach().numpy())
            outputs+= list(output)

        return targets, np.array(outputs), MCtrue, MCfalse

        
    def plot_loss(self, early=True):
        train_loss = self.train_output['train_loss']
        val_loss = self.train_output['val_loss']

        plt.style.use("kit")
        plt.figure(figsize=(8,6))
        n_epochs = len(train_loss)
        minloss = min(val_loss)
        plt.plot(range(1,len(train_loss)+1),train_loss, label='Training Loss', marker='None')
        plt.plot(range(1,len(val_loss)+1),val_loss,label='Validation Loss', marker='None')
        plt.plot([], [], ' ', label=f'best = {minloss:.4f}')

        # find position of lowest validation loss
        self.minposs = val_loss.index(min(val_loss))+1 
        if self.minposs != n_epochs and early:
            plt.axvline(self.minposs, ymax=0.8, linestyle=':', color='black',label='Early Stopping')

        watermark(scale=1.4, information=self.infos)
        plt.title(self.title, loc='right', color='#666666')
        plt.legend(loc='center right', frameon = True, framealpha = 1, facecolor = 'white', edgecolor = 'white')
        plt.xlabel('epochs')
        plt.ylabel('loss')
        plt.savefig(f'img/3_loss_{self.name}.pdf', bbox_inches='tight')
        plt.show()
        
    def plot_acc(self, early=True):
        train_acc = self.train_output['train_acc']
        val_acc = self.train_output['val_acc']

        plt.style.use("kit")
        n_epochs = len(train_acc)
        maxacc = max(val_acc)
        plt.plot(range(1,len(train_acc)+1),train_acc, label='Training Accuracy', marker='None')
        plt.plot(range(1,len(val_acc)+1),val_acc,label='Validation Accuracy', marker='None')
        plt.plot([], [], ' ', label=f'best = {maxacc:.4f}')
        
        # find position of lowest validation loss
        if self.minposs != n_epochs and early:
            plt.axvline(self.minposs, ymax=0.8, linestyle=':', color='black',label='Early Stopping')

        plt.xlabel('epochs')
        plt.ylabel('accuracy')
        watermark(scale=1.4, information=self.infos)
        plt.title(self.title, loc='right', color='#666666')
        plt.legend(loc='lower right', frameon = True, framealpha = 1, facecolor = 'white', edgecolor = 'white')
        plt.savefig(f'img/3_acc_{self.name}.pdf', bbox_inches='tight')
        plt.show()
        
    def plot_GNNoutput(self, bins=30, scale=1.6):
        plt.style.use("kit_hist")
        hist = plt.hist([self.MCtrue_output, self.MCfalse_output], bins=bins, histtype='stepfilled', facecolor='white',stacked=True, label=['MC true edge', 'MC false edge'])
        all_output = np.concatenate([self.MCtrue_output, self.MCfalse_output])
        plt.style.use("kit")
        hist_full = plt.hist(all_output, bins=bins, histtype='step', lw=1.5, facecolor=(0,0,0,0), hatch=None)
        plt.yscale('log')
#         plt.ylim(top=10e5)
        watermark(scale=1.4, information=self.infos, information2=self.infos2)
        plt.title(self.title, loc='right', color='#666666')
        plt.xlabel('GNN output')
        binwidth = np.mean(np.diff(hist[1]))
        plt.ylabel(f'Entries / ({binwidth:.2f})')
        plt.legend(bbox_to_anchor=(0.9,0.45), loc="center right")
        plt.savefig(f'img/3_GNNoutput_{self.name}.pdf', bbox_inches='tight')
        plt.show()

    def get_roc(self):
        fpr, tpr, thresholds = roc_curve(self.target, self.output)
        auc_val=auc(fpr, tpr)
        return fpr, tpr, auc_val
        
    def plot_roc(self):
        
        fpr, tpr, auc_value = self.get_roc()
        
        plt.style.use("kit")
        plt.plot(fpr, tpr, label='AUC = {1:.4f}'.format('$p$',auc_value), marker='None')
        watermark(scale=1.4, information=self.infos, information2=self.infos2)
        plt.title(self.title, loc='right', color='#666666')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend(loc='lower right')
        plt.savefig(f'img/3_roc_{self.name}.pdf', bbox_inches='tight')
        plt.show()
        
    def plot_confm(self, cut):
        plt.style.use("kit")
        output_conf = (self.output>cut)*1
        confm = confusion_matrix(self.target,output_conf, normalize='all')
        dispConf = ConfusionMatrixDisplay(confusion_matrix=confm, display_labels=['False', 'True'])
        dispConf.plot()
        plt.xlabel(r'Predicted label with $\delta$ = ' + f'{cut:.3f}')
        plt.ylabel('MC True label')
        plt.yticks(rotation = 90)
        plt.title(self.title, loc='right', color='#666666')
        plt.savefig(f'img/3_cm_{self.name}.pdf', bbox_inches='tight')
        plt.show()
 
        
    def get_metrics(self):

        MCtrue = np.array(self.MCtrue_output, dtype=np.float32)
        MCfalse = np.array(self.MCfalse_output, dtype=np.float32)

        P = len(MCtrue) 
        N = len(MCfalse)

        Ntotal = N+P #number of hits

        cuts = np.linspace(0.01, 0.99, self.ncuts)
        purity = [] #positive predictive value/ precision
        efficiency = [] #true positive rate, sensitivity    
        TNR = [] #correct classified false edges
        FNR = [] #incorrectly classified false edges

        for c in tqdm(cuts):
            FP = len(MCfalse[MCfalse>c]) #false positive
            TP = len(MCtrue[MCtrue>c]) #true positive
            purity.append(TP/(TP+FP))
            efficiency.append(TP/P)

            TN = len(MCfalse[MCfalse<=c])
            FN = len(MCtrue[MCtrue<=c])
            TNR.append(TN/N)
            FNR.append(FN/P)
            
        cutPos =  np.argmin(np.abs((np.array(purity)-np.array(efficiency))))
        best_cut = cuts[cutPos]
        
        return purity, efficiency, cuts, cutPos, TNR, FNR
        
    def plot_purity_efficiency(self, cuts, cut_pos, purity, efficiency, TNR, FNR, variable=None, xname='threshold', yname=None, save_name=None, loc='upper right'):           
        plt.style.use("kit") 
        
        plt.plot(cuts, purity, label='purity', marker='None')
        plt.plot(cuts, efficiency, label='efficiency', marker='None')
        if cut_pos:
            plt.axvline(cuts[cut_pos], ymax=0.7, linestyle=':', color='black',label=f'{variable} = {cuts[cut_pos]:.3f}')
            plt.plot([], [], ' ', label=f'pur = {purity[cut_pos]:.3f}')
            plt.plot([], [], ' ', label=f'eff = {efficiency[cut_pos]:.3f}')
        
        watermark(scale=1.4, information=self.infos, information2=self.infos2)
        plt.title(self.title, loc='right', color='#666666')
        plt.xlabel(xname)
        if yname:
            plt.ylabel(yname)
        plt.legend(loc=loc, frameon = True, framealpha = 0.8, facecolor = 'white', edgecolor = 'white')
        plt.savefig(save_name, bbox_inches='tight')
        plt.show() 
        
        print(f'best {variable} threshold at {cuts[cut_pos]:.4f}, removed bad (TNR): {TNR[cut_pos]:.3f}, lost good (FNR): {FNR[cut_pos]:.3f}')
         
    def plot_metrics(self, output_bins=30, scale=1.6):      
        self.plot_GNNoutput(output_bins, scale)
        self.plot_roc()
        purity, efficiency, cuts, cutPos, TNR, FNR = self.get_metrics()
        self.plot_purity_efficiency(cuts, cutPos, purity, efficiency, TNR, FNR, variable=r'$\delta$', xname=r'$\delta$', yname='purity & efficiency', save_name=f'img/3_edge_weight_pureff_{self.name}.pdf', loc='lower left')
        self.plot_confm(cut=cuts[cutPos])