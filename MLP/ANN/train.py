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
from nn.model import ANN, MultitaskNN, RestNN
from datasets.timeseries import TimeSeriesDataset
from torch.utils.data import DataLoader
from typing import Optional
from tqdm.auto import tqdm
from eval.validation import *
from tqdm.auto import tqdm 
from util.earlystop import EarlyStopper
#from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingWarmRestarts
import warnings
warnings.filterwarnings('ignore')

torch.manual_seed(42)

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
        if X.dim() == 3:
            X, y = X.flatten(1), y[:,:,-1] 
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
            if X.dim() == 3:
                X, y = X.flatten(1), y[:,:,-1] 
            output = model(X)
            total_loss += criterion(output, y).item() * len(y)
            if metric is not None:
                output = torch.round(output)
                metric.update_state(output, y)

    total_loss = total_loss/len(data_loader.dataset)
    return total_loss 

def predict(model:nn.Module, dl:torch.utils.data.DataLoader, device) -> np.array:
    with torch.inference_mode():
        for x in dl:
            x = x[0].to(device)
            if x.dim() == 3:
                x = x.flatten(1)
            out = model(x).detach().cpu().numpy()
        out = np.concatenate([out[:,0], out[-1,1:]])

    return out

def dynamic_predict(model:nn.Module, t_data:TimeSeriesDataset, params:dict) -> list:
    pred = []
    x,out = t_data[len(t_data)+params['pred_size']-1]
    model = model.cpu()
    if x.ndim >= 2:
        input_size = params['input_size']*params['channel_num']
        #out = out.reshape(-1)
        #x = x.reshape(-1)
    else:
        input_size = params['input_size']
    with torch.inference_mode():
        for _ in range(int(params['tst_size']/params['pred_size'])):
            x = x.reshape(-1)
            out = out.reshape(-1)
            x = np.concatenate([x,out],dtype=np.float32)[-input_size:]
            x = torch.tensor(x).cpu()
            x = x.reshape(1,-1)
            out = model(x).detach().cpu().numpy()
            pred.append(out) 
        pred = np.concatenate(pred)
        
    return pred

def main(args):
    train_params = args.get("train_params")
    files_ = args.get("files")
    device = torch.device(train_params.get("device"))
    params = args.get("model_params")
    dl_params = train_params.get("data_loader_params")
    
    df = pd.read_csv("../../../../../estsoft/data/train.csv")
    df['datetime'] = pd.to_datetime(df['datetime'])

    ## make time-series dataset for input data
    df_pc, df_main = get_time_series(df)
    real = df_main[-params['tst_size']:].values

    trn_pc = TimeSeriesDataset(df_pc.to_numpy(dtype=np.float32)[:-params['tst_size']], 
                                 params['input_size'], params['pred_size'])
    trn_main = TimeSeriesDataset(df_main.to_numpy(dtype=np.float32)[:-params['tst_size']], 
                                 params['input_size'], params['pred_size'])
    tst_pc = TimeSeriesDataset(df_pc.to_numpy(dtype=np.float32)[-params['tst_size']-params['input_size']:], 
                                 params['input_size'], params['pred_size'])
    tst_main = TimeSeriesDataset(df_main.to_numpy(dtype=np.float32)[-params['tst_size']-params['input_size']:], 
                                 params['input_size'], params['pred_size'])
 
    trn_pc_dl = torch.utils.data.DataLoader(trn_pc, batch_size=dl_params.get("batch_size"), 
                                           shuffle=dl_params.get("shuffle"))
    tst_pc_dl = torch.utils.data.DataLoader(tst_pc, batch_size=params['tst_size'], shuffle=False)
    trn_main_dl = torch.utils.data.DataLoader(trn_main, batch_size=dl_params.get("batch_size"), 
                                           shuffle=dl_params.get("shuffle"))
    tst_main_dl = torch.utils.data.DataLoader(tst_main, batch_size=params['tst_size'], shuffle=False)
    nets = {
        "multi_channel":[],
        "nomal":[],
        "resnet":[]
    }
    if args.get("nomal"):
        nets['nomal'] = (ANN(params['input_size'],params['pred_size'],params['hidden_dim']).to(device))
    if args.get("multi"):
        nets['multi_channel'] = (MultitaskNN(params['input_size'],params['pred_size'],params['hidden_dim'],3).to(device))
    if args.get("resnet"):
        nets['resnet'] = (RestNN(params['input_size'],params['pred_size'],params['hidden_dim']).to(device))
    print(nets.values())
    
    history = {
        'trn_loss':[],
        'val_loss':[],
        'lr':[]
    }
    
    if args.get("train"):
        pbar = range(train_params.get("epochs"))
    
    print("Learning Start!")
    for i,net in enumerate(nets):
        if nets[net] and net in ['nomal','resnet']:
            trn_dl = trn_main_dl
            tst_dl = tst_main_dl
            tst = trn_main
        elif nets[net] and net == 'multi_channel':
            trn_dl = trn_pc_dl
            tst_dl = tst_pc_dl
            tst = trn_pc
        else:
            continue
        print('Task{} {}!'.format(i+1,net))
        early_stopper = EarlyStopper(train_params.get("patience") ,train_params.get("min_delta"))
        optim = torch.optim.AdamW(nets[net].parameters(), lr=train_params.get('optim_params').get('lr'))
        scheduler = CosineAnnealingWarmRestarts(optim, T_0=20, T_mult=1, eta_min=0.00000001)
        if train_params.get("pbar"):
            pbar = tqdm(pbar)
        for _ in pbar:
            
            loss = train(nets[net], nn.MSELoss(), optim, trn_dl, device)
            history['lr'].append(optim.param_groups[0]['lr'])
            if args.get("scheduler"):
                scheduler.step(loss)
            history['trn_loss'].append(loss)
            loss_val = evaluate(nets[net], nn.MSELoss(), tst_dl, device) 
            history['val_loss'].append(loss_val)
            pbar.set_postfix(trn_loss=loss,val_loss=loss_val)
            if early_stopper.early_stop(nets[net], loss_val, files_.get("output")+files_.get("name")+'_earlystop.pth', 
                                        train_params.get("early_stop")):           
                break
            
        nets[net].eval()
        out = predict(nets[net], tst_dl, device)

        get_graph(history, files_.get("name")) 
        pred = out.flatten()
        pred = torch.tensor(pred)
        print("Done!")
        torch.save(nets[net].state_dict(), files_.get("output")+str(i)+files_.get("name")+'.pth')
        if torch.load(files_.get("output")+files_.get("name")+'_earlystop.pth'):
            nets[net].load_state_dict(torch.load(files_.get("output")+files_.get("name")+'_earlystop.pth'))
        else:
            nets[net].load_state_dict(torch.load(files_.get("output")+str(i)+files_.get("name")+'.pth'))
        
        nets[net].eval()
            
        # visualization model's performance
        out_dynamic = dynamic_predict(nets[net], tst, params)
        out_dynamic = np.concatenate(out_dynamic)
        pred_dynamic = torch.tensor(out_dynamic)
        val_score = metric(pred, real)
        pred_score = metric(pred_dynamic, real)
        print('val score : ',val_score)
        print('pred score : ',pred_score)
        get_r2_graph(pred_dynamic, real, pred, real, str(i)+'_'+files_.get("name"))
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
