import pandas as pd
import numpy as np
from datasets.preprocess import fill_null, make_target

def get_time_series(df):
    df_prod =df[df['is_consumption']==0]
    df_cons = df[df['is_consumption']==1]
     
    df_prod = df_prod[df_prod['prediction_unit_id']==0]
    df_cons = df_cons[df_cons['prediction_unit_id']==0]
    
    # custom preprocessing
    df_prod = fill_null(df_prod)
    df_cons = fill_null(df_cons)
    '''
    # make main dataframe
    df_main = df_prod.set_index('datetime').join(df_cons.set_index('datetime'), lsuffix='_prod', rsuffix='_cons')
    df_main['main_target'] = main_target
    #print(df_main.columns)
    '''
    df_main = make_target(df_prod, df_cons)
    
    #df_target = df_main.main_target
    df_prod = np.array(df_prod['target'])
    df_cons = np.array(df_cons['target'])
    df_total = np.concatenate((df_prod,df_cons))
    df_total = np.concatenate((df_total,df_main)).reshape(3,15312)

    #assert df_main.isna().sum()==False, 'Null value exists in prod'
    return df_total, df_main