import numpy as np
import torch

class TimeSeriesDataset(torch.utils.data.Dataset):
  def __init__(self, ts:np.array, window_size:int, forecast_size:int, d:int=0, m:int=1):
    self.d, self.m = d, m
    self.window_size, self.forecast_size = window_size, forecast_size
    self._data = ts

    new_ts = ts
    for i in range(d):
      new_ts = new_ts[m:] - new_ts[:-m]
    self.data = new_ts

  def __len__(self):
    return len(self.data) - self.window_size - self.forecast_size + 1

  def __getitem__(self, i):
    x = self.data[i:(i+self.window_size)]
    y = self.data[(i+self.window_size):(i+self.window_size+self.forecast_size)]
    return x, y