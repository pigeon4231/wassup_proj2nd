import pandas as pd
import torch
import torchmetrics
import numpy as np
import os
import glob
import argparse

from torch import nn
from datasets.dataset import get_time_series
from metric.metric import metric
from metric.visualization import get_r2_graph, get_graph
from nn.model import ANN
from datasets.timeseries import TimeSeriesDataset
from torch.utils.data import DataLoader
from typing import Optional
from tqdm.auto import tqdm
from eval.validation import *
from tqdm.auto import tqdm 
#from nn.early_stop import EarlyStopper
#from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingWarmRestarts

import warnings
warnings.filterwarnings('ignore')

device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

def train(
    model:nn.Module,
    criterion:callable,
    optimizer:torch.optim.Optimizer,
    data_loader:DataLoader,
    device:str
) -> float:
    '''train one epoch
    
    Args:
        model: model
        criterion: loss
        optimizer: optimizer
        data_loader: data loader
        device: device
    '''
    model.train()
    total_loss = 0.
    for X, y in data_loader:
        X, y = X.to(device), y.to(device)
        output = model(X)
        loss = criterion(output, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(y)
    return total_loss/len(data_loader.dataset)

def evaluate(
    model:nn.Module,
    criterion:callable,
    data_loader:DataLoader,
    device:str,
    metric:Optional[torchmetrics.metric.Metric]=None,
) -> float:
    '''evaluate
    
    Args:
        model: model
        criterions: list of criterion functions
        data_loader: data loader
        device: device
    '''
    model.eval()
    total_loss = 0.
    with torch.inference_mode():
        for X, y in data_loader:
            X, y = X.to(device), y.to(device)
            output = model(X)
            total_loss += criterion(output, y).item() * len(y)
            if metric is not None:
                output = torch.round(output)
                metric.update_state(output, y)

    total_loss = total_loss/len(data_loader.dataset)
    return total_loss 

def main(args):
    tst_size = 96
    
    train_params = args.get("train_params")
    files_ = args.get("files")
    device = torch.device(train_params.get("device"))
    model_params = args.get("model_params")
    dl_params = train_params.get("data_loader_params")
    
    df = pd.read_csv("../../../../../estsoft/data/train.csv")
    df['datetime'] = pd.to_datetime(df['datetime'])

    ## make time-series dataset for input data
    df_prod, df_cons = get_time_series(df)
    a = df_prod[-tst_size:]
    trn_prod = TimeSeriesDataset(df_prod.to_numpy(dtype=np.float32)[:-tst_size], 26, 1)
    tst_prod = TimeSeriesDataset(df_prod.to_numpy(dtype=np.float32)[-tst_size-26:], 26, 1)
    trn_cons = TimeSeriesDataset(df_cons.to_numpy(dtype=np.float32)[:-tst_size], 26, 1)
    tst_cons = TimeSeriesDataset(df_cons.to_numpy(dtype=np.float32)[-tst_size-26:], 26, 1)
    trn_prod_dl = torch.utils.data.DataLoader(trn_prod, batch_size=dl_params.get("batch_size"), 
                                           shuffle=dl_params.get("shuffle"))
    tst_prod_dl = torch.utils.data.DataLoader(tst_prod, batch_size=96, shuffle=False)
    trn_cons_dl = torch.utils.data.DataLoader(trn_cons, batch_size=dl_params.get("batch_size"), 
                                           shuffle=dl_params.get("shuffle"))
    tst_cons_dl = torch.utils.data.DataLoader(tst_cons, batch_size=96, shuffle=False)
    
    net = ANN(26,1,256).to(device)
    print(net)
    
    optim = torch.optim.AdamW(net.parameters(), lr=0.00001)
    #optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    #scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=1, eta_min=0.00001)
  
    history = {
        'trn_loss':[],
        'val_loss':[],
        'lr':[]
    }
    print('learning start!')
    print('Task 1')
    if args.get("train"):
        pbar = range(train_params.get("epochs"))
    if train_params.get("pbar"):
        pbar = tqdm(pbar)
    
    print("Learning Start!")
    #early_stopper = EarlyStopper(args.patience ,args.min_delta)
    for _ in pbar:
        loss = train(net, nn.MSELoss(), optim, trn_prod_dl, device)
        history['lr'].append(optim.param_groups[0]['lr'])
        #scheduler.step(loss)
        history['trn_loss'].append(loss)
        loss_val = evaluate(net, nn.MSELoss(), tst_prod_dl, device) 
        history['val_loss'].append(loss_val)
        pbar.set_postfix(trn_loss=loss,val_loss=loss_val)
        #if early_stopper.early_stop(model, loss, args.output+args.name+'_earlystop.pth'):
            #print('Early Stopper run!')            
            #break
    
    net.eval()
    pred = []
     
    with torch.inference_mode():
        for x in tst_prod_dl:
            x = x[0].cpu()
            out = net(x)

    get_graph(history, files_.get("name")) 
    pred_prod = out
    pred_cons = out
 
    print("Done!")
    torch.save(net.state_dict(), files_.get("output")+files_.get("name")+'.pth')
    # visualization model's performance
    get_r2_graph(pred_prod, a.values, pred_cons, a.values, files_.get("name"))
    pred_prod = torch.tensor(pred_prod)
    pred_cons = torch.tensor(pred_cons)
    score_pr = metric(pred_prod, a.values)
    score_co = metric(pred_cons, a.values)
    print('prod score : ',score_pr)
    print('cons score : ',score_co)

    print('------------------------------------------------------------------')
    if args.get("validation"):
        model = ANN(X_trn.shape[-1] ,model_params.get("hidden_dim")).to(device)
        scores = Validation(X_trn, y_trn, train_params.get("patience"), train_params.get("min_delta"))
        scores = pd.DataFrame(scores.kfold(model, n_splits=5, epochs=train_params.get("epochs"), lr=opt_params.get("lr"), 
                                        batch=dl_params.get("batch_size"), shuffle=True, random_state=2023))
        print(pd.concat([scores, scores.apply(['mean', 'std'])]))
        
        return

def get_args_parser(add_help=True):
    parser = argparse.ArgumentParser(description="Pytorch K-fold Cross Validation", add_help=add_help)
    parser.add_argument(
        "-c", "--config", default="./configs/config.py", type=str, help="configuration file"
    )
    parser.add_argument(
        "-mode", "--multi-mode", default=False, type=bool, help="multi train mode"
    )

    return parser

if __name__ == "__main__":
    args = get_args_parser().parse_args()
    print(args.config)
    exec(open(args.config).read())
    main(config)
    
    if args.multi_mode:
        for filename in glob.glob("config/multi*.py"):
            print(filename)
            exec(open('./'+filename).read())
            main(config)
