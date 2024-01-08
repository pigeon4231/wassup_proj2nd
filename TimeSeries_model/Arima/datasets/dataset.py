import pandas as pd
from datasets.preprocess import fill_null

def get_time_series(df):
    tst_size = 96
    df_prod =df[df['is_consumption']==0]
    df_cons = df[df['is_consumption']==1]

    time_before = ['2022-10-30 02:00:00', '2022-03-27 02:00:00', '2023-03-26 02:00:00', '2021-10-31 02:00:00']
    time = ['2022-10-30 03:00:00', '2022-03-27 03:00:00', '2023-03-26 03:00:00', '2021-10-31 03:00:00']
    time_after = ['2022-10-30 04:00:00', '2022-03-27 04:00:00', '2023-03-26 04:00:00', '2021-10-31 04:00:00']

    df_prod = fill_null(df_prod, time_before, time, time_after)
    df_cons = fill_null(df_cons, time_before, time, time_after)
     
    df_prod = df_prod[df_prod['prediction_unit_id']==0]
    df_cons = df_cons[df_cons['prediction_unit_id']==0]
    
    df_prod = df_prod.set_index('datetime')
    df_cons = df_cons.set_index('datetime')
    
    trn_pr, tst_pr = df_prod[:-tst_size], df_prod[-tst_size:]
    trn_cn, tst_cn = df_cons[:-tst_size], df_cons[-tst_size:]
    
    
    assert df_prod.target.isna().sum()==False, 'Null value exists in prod'
    assert df_cons.target.isna().sum()==False, 'Null value exists in cons'
    return trn_pr, tst_pr, trn_cn, tst_cn