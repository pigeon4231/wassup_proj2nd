import numpy as np
import torch

class TimeSeriesDataset(torch.utils.data.Dataset):
  def __init__(self, ts:np.array, patch_size:int=4, n_token:int=6):
    self.patch_size = patch_size
    self.n_patch = 4
    self.n_token = n_token
    self.window_size = int(patch_size * self.n_patch * n_token / 2) # 48
    self.forecast_size = patch_size
    self.data = ts

  def __len__(self):
    return int((len(self.data) - self.window_size - self.forecast_size)/self.forecast_size) + 1

  def __getitem__(self, i):
    look_back = self.data[i:(i+self.window_size)] # 48
    look_back = np.concatenate([look_back] + [look_back[-self.patch_size:]] * int(self.n_patch / 2)) # [1,2,...,48] + [45,46,47,48]*2  = 56
    x = np.array([look_back[i*int(self.patch_size*self.n_patch/2):(i+2)*int(self.patch_size*self.n_patch/2)] for i in range(self.n_token)])
    #[0:16],[8:24],[16:32],[24:40],[32:48],[40:56]
    y = self.data[(i+self.window_size):(i+self.window_size+self.forecast_size)]
    #[48:52] = 4
    return x, y