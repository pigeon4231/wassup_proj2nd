import pandas as pd
import numpy as np
from datasets.preprocess import fill_null

def get_time_series(df):
    df_prod =df[df['is_consumption']==0]
    df_cons = df[df['is_consumption']==1]
     
    df_prod = df_prod[df_prod['prediction_unit_id']==0]
    df_cons = df_cons[df_cons['prediction_unit_id']==0]
    
    # custom preprocessing
    df_prod = fill_null(df_prod)
    df_cons = fill_null(df_cons)
    df_prod = df_prod.target
    df_cons = df_cons.target   
    
    assert df_prod.isna().sum()==False, 'Null value exists in prod'
    assert df_cons.isna().sum()==False, 'Null value exists in cons'
    return df_prod, df_cons