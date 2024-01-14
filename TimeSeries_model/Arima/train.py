import pandas as pd
import torch
from statsmodels.tsa.arima.model import ARIMA 
from datasets.dataset import get_time_series
from metric.metric import metric
from metric.visualization import get_r2_graph

import warnings
warnings.filterwarnings('ignore')

name = 'test'

df = pd.read_csv("../../../../../estsoft/data/train.csv")
df['datetime'] = pd.to_datetime(df['datetime'])

# make time-series dataset for input data
trn_prod, tst_prod, trn_cons, tst_cons = get_time_series(df)
trn_prod = trn_prod.target.asfreq(freq='h')
trn_cons = trn_cons.target.asfreq(freq='h')
tst_prod = tst_prod.target.to_numpy().flatten()
tst_cons = tst_cons.target.to_numpy().flatten()

print('learning start!')
print('Task 1')
arimax_prod = ARIMA(trn_prod ,order=(26, 0, 0),).fit()
print('Task 2')
arimax_cons = ARIMA(trn_cons ,order=(26, 0, 0),).fit()
print('learning end!')

# model save
#arimax_prod.save('arima_prod_{}.pkl'.format(name))
#arimax_cons.save('arima_cons_{}.pkl'.format(name))

# prediction using model maked
pred_prod = arimax_prod.predict(
    start='2023-05-28 00:00:00', 
    end='2023-05-31 23:00:00',
    dinamic=False)
pred_cons = arimax_cons.predict(
    start='2023-05-28 00:00:00', 
    end='2023-05-31 23:00:00',
    dinamic=False)

# visualization model's performance
get_r2_graph(pred_prod, tst_prod, pred_cons, tst_cons, name)
pred_prod = torch.tensor(pred_prod)
pred_cons = torch.tensor(pred_cons)
score_pr = metric(pred_prod, tst_prod)
score_co = metric(pred_cons, tst_cons)
print('prod score : ',score_pr)
print('cons score : ',score_co)
