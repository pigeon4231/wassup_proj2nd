import torch.nn as nn
import torch

class ANN(nn.Module):
  '''
  function for building Neural network 
  
  this neural network is composed two hidden layer
  
  args:
    input: int
    hidden: int
    
  '''
  def __init__(self, input:int=26, output:int=1, hidden:int=64):
    super().__init__()
    self.linear_stack = nn.Sequential(
        nn.Linear(input,hidden),      #(18,32)
        nn.BatchNorm1d(hidden),
        nn.ELU(),
        #nn.Dropout(0.3),
        nn.Linear(hidden,hidden*2),   #(32,64)
        nn.BatchNorm1d(hidden*2),
        nn.ELU(),
        #nn.Dropout(0.3),
        nn.Linear(hidden*2,hidden*2),   #(32,64)
        nn.BatchNorm1d(hidden*2),
        nn.ELU(),
        #nn.Dropout(0.3),     
        #nn.Dropout(0.3),
        nn.Linear(hidden*2,output),        #(128,1)
        )
    
  def forward(self, x:torch.Tensor):
    x = self.linear_stack(x)
    return x
  
class MultitaskNN(nn.Module):
  '''
  function for building Neural network 

  this neural network is composed two hidden layer

  args:
    input: int
    hidden: int
    
  '''
  def __init__(self, input:int=26, output:int=2, hidden:int=64, channel_num=3):
    super().__init__()
    self.linear_stack = nn.Sequential(
        nn.Linear(input*channel_num, hidden),      #(18,32)
        nn.BatchNorm1d(hidden),
        nn.ELU(),
        #nn.Dropout(0.3),
        nn.Linear(hidden,hidden*2),   #(32,64)
        nn.BatchNorm1d(hidden*2),
        nn.ELU(),
        #nn.Dropout(0.3),
        nn.Linear(hidden*2,hidden*2),   #(32,64)
        nn.BatchNorm1d(hidden*2),
        nn.ELU(),
        #nn.Dropout(0.3),     
        #nn.Dropout(0.3),
        nn.Linear(hidden*2,output),        #(128,1)
        )
    
  def forward(self, x:torch.Tensor):
    x = self.linear_stack(x)
    return x
  
class RestNN(nn.Module):
  '''
  function for building Neural network 
  
  this neural network is composed two hidden layer
  
  args:
    input: int
    hidden: int
    
  '''
  def __init__(self, input:int=26, output:int=1, hidden:int=64):
    super().__init__()
    self.linear_stack = nn.Sequential(
      nn.Linear(input,hidden),      #(18,32)
      nn.BatchNorm1d(hidden),
      nn.ELU(),
      #nn.Dropout(0.3),
      nn.Linear(hidden,hidden*2),   #(32,64)
      nn.BatchNorm1d(hidden*2),
      nn.ELU(),
      #nn.Dropout(0.3),
      nn.Linear(hidden*2,hidden*2),   #(32,64)
      nn.BatchNorm1d(hidden*2),
      nn.ELU(),
      #nn.Dropout(0.3),     
      #nn.Dropout(0.3),
      nn.Linear(hidden*2,input),        #(128,1)
      nn.ELU()
      )
    self.res_stack = nn.Sequential(
      nn.Linear(input,hidden),
      nn.BatchNorm1d(hidden),
      nn.ELU(),
      nn.Linear(hidden,hidden*2),
      nn.BatchNorm1d(hidden*2),
      nn.ELU(),
      nn.Linear(hidden*2,input),
      nn.ELU()
    )
    self.fc_layer = nn.Sequential(
      nn.Linear(input,output)
    )
    
  def forward(self, x:torch.Tensor):
    x_ = self.linear_stack(x)
    x = self.res_stack(x-x_)
    x = self.fc_layer(x_+x)
    return x