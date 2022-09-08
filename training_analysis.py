import os
import shutil
import numpy as np
from time import time
from tqdm import tqdm
import copy
import pandas as pd

import resource
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (2048, rlimit[1]))

import torch
import torch_geometric
torch.cuda.is_available()
from torch_geometric.loader import DataLoader
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import torchmetrics
import torch.nn as nn
import torch.nn.functional as F
# torch.manual_seed(42)

from utils.data.build_graph import build_graphs
from utils.data.load_graphs import load_graphs
from utils.data.graphdata import GraphDataset
from utils.models.interaction_network import InteractionNetwork
# from utils.training.train_3 import train_model
from utils.training.pytorchtools import EarlyStopping
from utils.evaluation.evaluate_model import evaluate_model

import optuna

def train(epoch, model, optimizer, train_loader):

    model.train()
    losses = []
    epoch_t0 = time()
    acc = torchmetrics.Accuracy().to(device)

    for batch_idx, data in enumerate(train_loader):
        data = data.to(device) 
        if (len(data.x)==0): continue        
        target = data.y
        optimizer.zero_grad()
        output = model(data).squeeze(1)
        loss = F.binary_cross_entropy(output,target)
        loss.backward()
        optimizer.step()
        acc.update(output, target.int())
        losses.append(loss.item())      

    print(f'Train Epoch: {epoch}, epoch time: {time()-epoch_t0:.2f}s')
    print(f'Mean Train Batch Loss: {np.nanmean(losses):.04f}')

    return np.nanmean(losses), acc.compute()

    
def validate(model, val_loader, threshold=0.5):
    model.eval()
    val_loss = 0
    val_acc = torchmetrics.Accuracy(threshold).to(device)
    val_losses = []
    with torch.no_grad():                # disableing gradient calculation for inference 
        for data in val_loader:
            data = data.to(device)
            if (len(data.x)==0): continue 
            target = data.y
            output = model(data).squeeze(1)
            val_loss = F.binary_cross_entropy(output,target)
            val_acc.update(output, target.int())
            val_losses.append(val_loss.item())

    print('\nTest set: Average loss: {:.4f}\n, Accuracy: {:.4f}\n'
            .format(np.mean(val_losses), val_acc.compute()))
    return np.nanmean(val_losses), val_acc.compute()



def get_model(trial):
    model = InteractionNetwork(node_dim, edge_dim, hidden_dim).to(device)

    return model, hidden_dim


def objective(trial):

    lr = trial.suggest_float("lr", 1e-3, 1e-1, log=True)
    gamma = trial.suggest_float('gamma', 0.6, 0.99)
    lr_step_size = trial.suggest_int('step_size', 1, 9)
    batch_size = 1
    patience = 5
    print(lr, gamma, lr_step_size)

    graphs = load_graphs(graph_dir, nevents, node_dim, edge_dim)

    parts = [0.8, 0.1, 0.1] #sizes of training, validation and testing samples
    load_params = {'batch_size': batch_size, 'shuffle': True, 'num_workers': 30}
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

    model, hidden_dim = get_model(trial)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = StepLR(optimizer, step_size=lr_step_size, gamma=gamma)

    name = f'IN_{nevents}_{hidden_dim}_{lr}_{gamma}_{lr_step_size}'

    # initialize the early_stopping object
    early_stopping = EarlyStopping(patience=patience, verbose=True, path=f'models/checkpoint_{out_name}.pt')

    losses, accs = [], []
    val_losses, val_accs = [], []
    output = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    for epoch in tqdm(range(1, epochs + 1)):
        for i in range(iterations):
            train_loss, train_acc = train(epoch, model, optimizer, train_loader)
            losses.append(train_loss)
            accs.append(train_acc)

        val_loss, val_acc = validate(model, val_loader, threshold=0.5)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        scheduler.step()

        # early_stopping needs the validation loss to check if it has decreased, 
        # and if it has, it will make a checkpoint of the current model
        early_stopping(val_loss, model)

        if early_stopping.early_stop:
            print("Early stopping")
            break
            # load the last checkpoint with the best model
	
        trial.report(val_loss, epoch)

        # Handle pruning based on the intermediate value.
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

        output['train_loss'].append(train_loss)
        output['train_acc'].append(train_acc)
        output['val_loss'].append(val_loss)
        output['val_acc'].append(val_acc)        
    
    model.load_state_dict(torch.load(f'models/checkpoint_{out_name}.pt'))

    np.save(f'train_output/{out_name}/train_{name}', output)	    

    #save model
    trained_model = copy.deepcopy(model)

    torch.save(model.state_dict(), f"models/{out_name}/{name}_state_dict.pt")
    torch.save(model, f'models/{out_name}/{name}.pt')
    

    return np.min(val_losses)



if __name__ == "__main__":

    device = torch.device('cpu')
    epochs = 50
    # epochs = 20
    iterations = 3
    # iterations = 2
    n_trials = 20
    run = 'seg_1'
    minlayer = 0
    maxlayer = 8
    # nevents_range = [100, 400, 600, 1000]
    nevents_range = [1000]
    # nhidden_range = [3,4,5,6,7,8, 16, 32]
    nhidden_range = [9,10,11,12,13,14,15,16]
    pz_min = 0.001
    slope_max = 2.
    node_dim = 2
    edge_dim = 2


    
    out_name = f'optimization_{run}'       

    raw = pd.read_hdf('data/raw.h5', 'df')  
    graph_dir = f'data/graphs_{run}_pzmin_{pz_min}_slopemax_{slope_max}'

    data = build_graphs(raw, shuffle=True,  pz_min=pz_min, remove_duplicates=True, slope=slope_max, graph_dir=graph_dir)
    graphs = data.create_graph_list(node_dim=node_dim, edge_dim=edge_dim, minlayer=minlayer, maxlayer=maxlayer)

    os.makedirs(f'models/{out_name}', exist_ok=True)
    os.makedirs(f'models/{out_name}/best', exist_ok=True)
    os.makedirs(f'train_output/{out_name}', exist_ok=True)

    f = open(f'{out_name}.txt', 'a')
    f.write(f' nevents: {nevents_range}, nhidden: {nhidden_range}, pzmin: {pz_min}, slope_max: {slope_max}, node_dim: {node_dim}, edge_dim: {edge_dim}, min_layer: {minlayer}, max_layer: {maxlayer}\n ')
    f.close()


    for nevents in nevents_range:
         
        f = open(f'{out_name}.txt', 'a')
        f.write(f'\n=====================================================================================================================\n {nevents} events, hidden dims: {nhidden_range} \n =====================================================================================================================\n \n')
        f.close()

        for hidden_dim in nhidden_range:

            study = optuna.create_study(direction="minimize")
            study.optimize(objective, n_trials=n_trials, catch=(ValueError,))

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
            best_model = f'IN_{nevents}_{hidden_dim}'

            print("  Value: ", trial.value)

            print("  Params: ")
            for key, value in trial.params.items():
                print("    {}: {}".format(key, value))
                best_model += '_' + str(value)

            best_model_dict = best_model + '_state_dict.pt'
            best_model += '.pt'

            shutil.copy2(f'models/{out_name}/{best_model}', f'models/{out_name}/best/best_model_nevents_{nevents}_hidden_dim_{hidden_dim}.pt')
            shutil.copy2(f'models/{out_name}/{best_model_dict}', f'models/{out_name}/best/best_model_nevents_{nevents}_dict_hidden_dim_{hidden_dim}.pt')        


            f = open(f'{out_name}.txt', 'a')
            f.write(f'\n ---trial for hidden_dim: {hidden_dim} at {nevents} events, {epochs} epochs, {iterations} iterations, {n_trials} trials --- \n')
            f.write('best loss: ' + str(trial.value) + '\n')
            for key, value in trial.params.items():
                f.write(key)
                f.write(': ')
                f.write(f'{value}')
                f.write('\n')
            f.write(f'model saved under /models/{out_name}/best/best_model_hidden_dim_{hidden_dim}.pt \n \n')
            f.close()







