import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    
    def __init__(self,max_len,d_model,device):
        super(PositionalEncoding,self).__init__()
        
        self.encoding = torch.zeros(max_len,d_model,device = device)
        self.encoding.requires_grad = False # we don't need to compute gradient
        
        pos = torch.arange(0,max_len,device=device)
        pos = pos.float().unsqueeze(dim = 1)
        
        _2i = torch.arange(0,d_model,step = 2,device = device).float()
        
        self.encoding[:,0::2] = torch.sin(pos/(10000**(_2i/d_model)))
        self.encoding[:,1::2] = torch.cos(pos/(10000**(_2i/d_model)))
        
    def forward(self,x):
        batch_size,seq_len = x.size()
        
        return self.encoding[:seq_len,:]