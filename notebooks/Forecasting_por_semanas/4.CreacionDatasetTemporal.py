import sys
sys.path.insert(1, '../../Src')
from utils.preprocessing import *



import matplotlib.pyplot as plt
import os, sys, gc, time, warnings, pickle, psutil, random
from math import ceil
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
import numpy as np
import pandas as pd
import os

warnings.filterwarnings('ignore')


PATH_DATASET = '../../dataset/'
PATH_RESULTS = '../../results/Demanda/'

FILL = False
TARGET = 'Demanda'

SHIFT = 1

replace_nan = False
THRESHOLD_NAN = 0.6

df_test   = pd.read_pickle(os.path.join(PATH_RESULTS,'dataset','df_test.pkl'))
df_test.head()



print('Reading dataset')
df_train  = pd.read_pickle(os.path.join(PATH_RESULTS,'dataset','df_train.pkl'))
df_test   = pd.read_pickle(os.path.join(PATH_RESULTS,'dataset','df_test.pkl'))

df_train["item_id"] = df_train["Z_MODELO"].astype(str) +"|"+ df_train["Z_PUNTO_VENTA"].astype(str) +"|"+ df_train["Z_GAMA"].astype(str) 
df_test["item_id"] = df_test["Z_MODELO"].astype(str) +"|"+ df_test["Z_PUNTO_VENTA"].astype(str) +"|"+ df_test["Z_GAMA"].astype(str) 


df_release = pd.read_pickle(os.path.join(PATH_RESULTS,'dataset','grid_part_1.pkl'))
print('df_release size ',df_release.shape)

df_test.merge(df_release[['date_block_num','item_id','release']],on=['date_block_num','item_id'],how='left')
df_train.merge(df_release[['date_block_num','item_id','release']],on=['date_block_num','item_id'],how='left')
del df_release
gc.collect()

print('df_train size :',df_train.shape)
print('df_test  size :',df_test.shape)


# Add features 


df_train.replace([np.inf, -np.inf, np.nan],0,inplace=True)
df_test.replace([np.inf, -np.inf, np.nan],0,inplace=True)

gc.collect()
df_test[TARGET] = -1



############## dates ##################################

df_fe_dates = pd.read_pickle(os.path.join(PATH_RESULTS,'dataset','grid_part_2.pkl'))

print('df_fe_dates size :',df_fe_dates.shape)

df_train = df_train.merge(df_fe_dates,on=['item_id','date_block_num'],how='left')
df_test  = df_test.merge(df_fe_dates,on=['item_id','date_block_num'],how='left')

if replace_nan:
    print('replace_nan',replace_nan)
    df_train.replace([np.inf, -np.inf, np.nan],0,inplace=True)
    df_test.replace([np.inf, -np.inf, np.nan],0,inplace=True)

del df_fe_dates
gc.collect()
print('df_train size:',df_train.shape)
print('df_test size :',df_test.shape)

df_fe_lags = pd.read_pickle(os.path.join(PATH_RESULTS,'dataset','lags_df_'+str(SHIFT)+'_completed.pkl'))




############## LAGS ##################################

df_fe_lags = pd.read_pickle(os.path.join(PATH_RESULTS,'dataset','lags_df_'+str(SHIFT)+'_completed.pkl'))
del df_fe_lags[TARGET]
del df_fe_lags['Z_WEEK']
del df_fe_lags['Z_WEEK_DATE']

print('df_fe_lags size :',df_fe_lags.shape)


df_train = df_train.merge(df_fe_lags,on=['item_id','date_block_num'],how='left')
df_test  = df_test.merge(df_fe_lags,on=['item_id','date_block_num'],how='left')

if replace_nan:
    print('replace_nan',replace_nan)
    df_train.replace([np.inf, -np.inf, np.nan],0,inplace=True)
    df_test.replace([np.inf, -np.inf, np.nan],0,inplace=True)

del df_fe_lags
gc.collect()
print('df_train size:',df_train.shape)
print('df_test size :',df_test.shape)



from math import ceil

def week_of_month(dt):
    """ Returns the week of the month for the specified date.
    """

    first_day = dt.replace(day=1)

    dom = dt.day
    adjusted_dom = dom + first_day.weekday()

    return int(ceil(adjusted_dom/7.0))


df_train['week_of_month'] = df_train['Z_WEEK_DATE'].apply(week_of_month)
df_test['week_of_month']  = df_test['Z_WEEK_DATE'].apply(week_of_month)



############## Encodings ##################################

df_fe_encodings = pd.read_pickle(os.path.join(PATH_RESULTS,'dataset','mean_encoding_df.pkl'))

del df_fe_encodings['Z_WEEK_DATE']

print('df_fe_encodings size :',df_fe_encodings.shape)

df_train = df_train.merge(df_fe_encodings,on=['item_id','date_block_num'],how='left')
df_test  = df_test.merge(df_fe_encodings,on=['item_id','date_block_num'],how='left')


if replace_nan:
    print('replace_nan',replace_nan)
    df_train.replace([np.inf, -np.inf, np.nan],0,inplace=True)
    df_test.replace([np.inf, -np.inf, np.nan],0,inplace=True)


del df_fe_encodings
gc.collect()
#df_train = reduce_mem_usage(df_train)
#df_test  = reduce_mem_usage(df_test)
gc.collect()
print('df_train size             :',df_train.shape)
print('df_test size :',df_test.shape)


# Save Data

########################### Save part 1
#################################################################################
print('Save Dataset ..')

print('saving .. ',os.path.join(PATH_RESULTS,'dataset','df_train_fe_FILL_'+str(FILL)+'_SHIFT_'+str(SHIFT)+'.pkl'))
df_train.to_pickle(os.path.join(PATH_RESULTS,'dataset','df_train_fe_FILL_'+str(FILL)+'_SHIFT_'+str(SHIFT)+'.pkl'))

print('df_train Size:', df_train.shape)

print('saving .. ',os.path.join(PATH_RESULTS,'dataset','df_test_fe_FILL_'+str(FILL)+'_SHIFT_'+str(SHIFT)+'.pkl'))

df_test.to_pickle(os.path.join(PATH_RESULTS,'dataset','df_test_fe_FILL_'+str(FILL)+'_SHIFT_'+str(SHIFT)+'.pkl'))

print('df_test Size:', df_test.shape)