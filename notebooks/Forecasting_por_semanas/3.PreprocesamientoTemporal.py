import sys
sys.path.insert(1, '../../Src')
from utils.preprocessing import *


import matplotlib.pyplot as plt
import os, sys, gc, time, warnings, pickle, psutil, random
from sklearn.preprocessing import LabelEncoder
from joblib import Parallel, delayed
from math import ceil
from tqdm import tqdm
import seaborn as sns
import numpy as np
import pandas as pd
import time
import gc
import os
import warnings

SEED = 42
seed_everything(SEED)




SHIFT_DAY = 1
WINDOW = 15
TARGET = 'Demanda'         # Our main target
PATH_DATASET = '../../dataset/'
PATH_RESULTS = '../../results/Demanda/'


warnings.filterwarnings('ignore')
sns.set_theme(style="ticks", color_codes=True)
pd.set_option('display.max_columns', 100)
gc.collect()




df_train_original = pd.read_csv(os.path.join(PATH_DATASET,'train/train_converted.csv'))
df_test_original = pd.read_csv(os.path.join(PATH_DATASET,'test/test_converted.csv'))

df_train = df_train_original[['Z_MODELO','Z_PUNTO_VENTA','Z_GAMA','Z_WEEK','Z_WEEK_DATE','Demanda']].groupby(['Z_MODELO','Z_PUNTO_VENTA','Z_GAMA','Z_WEEK','Z_WEEK_DATE']).sum().reset_index()
df_test = df_test_original[['Z_MODELO','Z_PUNTO_VENTA','Z_GAMA','Z_WEEK','Z_WEEK_DATE','Demanda']].groupby(['Z_MODELO','Z_PUNTO_VENTA','Z_GAMA','Z_WEEK','Z_WEEK_DATE']).sum().reset_index()
df_train = df_train.merge(df_train_original[['Z_MODELO','Z_PUNTO_VENTA','Z_GAMA','Z_WEEK','Z_MARCA','Z_DEPARTAMENTO']],on=['Z_MODELO','Z_PUNTO_VENTA','Z_GAMA','Z_WEEK'],how='left')
df_test  = df_test.merge(df_test_original[['Z_MODELO','Z_PUNTO_VENTA','Z_GAMA','Z_WEEK','Z_MARCA','Z_DEPARTAMENTO']],on=['Z_MODELO','Z_PUNTO_VENTA','Z_GAMA','Z_WEEK'],how='left')


# Creacion de fecha datetime e identificador
df_train['Z_WEEK_DATE'] = pd.to_datetime(df_train['Z_WEEK_DATE'], errors='coerce')
df_test['Z_WEEK_DATE'] = pd.to_datetime(df_test['Z_WEEK_DATE'], errors='coerce')
dates = (set(df_train['Z_WEEK'].unique()) | set(df_test['Z_WEEK'].unique()))#df_auxiliar['Z_WEEK'].unique()
dates = sorted(dates)

dict_dates = {}
for idx,date in enumerate(dates):
    dict_dates[date] =idx
df_train['date_block_num'] = df_train['Z_WEEK'].map(dict_dates)
df_test['date_block_num'] = df_test['Z_WEEK'].map(dict_dates)
df_test['Demanda'] = 0

# Creacion de identificadores por modelo punto venta y gamma 

df_train["item_id"] = df_train["Z_MODELO"].astype(str) +"|"+ df_train["Z_PUNTO_VENTA"].astype(str) +"|"+ df_train["Z_GAMA"].astype(str) 
df_test["item_id"]  = df_test["Z_MODELO"].astype(str) +"|"+ df_test["Z_PUNTO_VENTA"].astype(str) +"|"+ df_test["Z_GAMA"].astype(str) 
df_train.fillna(-1, inplace=True)

df_train.to_pickle(os.path.join(PATH_RESULTS,'df_train.pkl'))

df_test.to_pickle(os.path.join(PATH_RESULTS,'df_test.pkl'))

# limpieza de memoria
del df_train
del df_test
del df_train_original
del df_test_original
del dict_dates
gc.collect()


# Feature Engineering with temporal information
## Loading dataset



df_train = pd.read_pickle(os.path.join(PATH_RESULTS,'dataset','df_train.pkl'))
df_test  = pd.read_pickle(os.path.join(PATH_RESULTS,'dataset','df_test.pkl'))


df_train.replace([np.inf, -np.inf, np.nan],0,inplace=True)
df_test.replace([np.inf, -np.inf, np.nan],0,inplace=True)

N_test  = df_test.shape[0]
N_train = df_train.shape[0]


df_train['DATE'] = pd.to_datetime(df_train['Z_WEEK_DATE'], errors='coerce')
df_test['DATE']  = pd.to_datetime(df_test['Z_WEEK_DATE'], errors='coerce')




# creacion de la data total training y testing para generar los features temporales
df_train_aux = pd.read_pickle(os.path.join(PATH_RESULTS,'dataset','df_train.pkl'))
df_test_aux  = pd.read_pickle(os.path.join(PATH_RESULTS,'dataset','df_test.pkl'))    

df_auxiliar = pd.concat([df_train_aux,df_test_aux])

df_auxiliar.replace([np.inf, -np.inf,np.nan],0 , inplace=True)
df_auxiliar.reset_index(inplace=True,drop=True)
print(df_auxiliar.shape)
df_auxiliar.head(2)

df_auxiliar['Z_WEEK_DATE']  = pd.to_datetime(df_auxiliar['Z_WEEK_DATE'], errors='coerce')

# limpieza de memoria
del df_train_aux
del df_test_aux
del df_train['Z_WEEK']
gc.collect()


# reducimos memoria
df_auxiliar = reduce_mem_usage(df_auxiliar)

# creamos el identificador para cada dataset
df_auxiliar["item_id"] = df_auxiliar["Z_MODELO"].astype(str) +"|"+ df_auxiliar["Z_PUNTO_VENTA"].astype(str) +"|"+ df_auxiliar["Z_GAMA"].astype(str) 

df_train["item_id"] = df_train["Z_MODELO"].astype(str) +"|"+ df_train["Z_PUNTO_VENTA"].astype(str) +"|"+ df_train["Z_GAMA"].astype(str) 
df_test["item_id"]  = df_test["Z_MODELO"].astype(str) +"|"+ df_test["Z_PUNTO_VENTA"].astype(str) +"|"+ df_test["Z_GAMA"].astype(str) 



# nos aseguramos que los datos esten ordenados para hacer los lags por tiempo
df_auxiliar.sort_values(by=['item_id','Z_WEEK_DATE'], ascending=[True, True],inplace=True)
df_test.sort_values(by=['item_id','Z_WEEK_DATE'], ascending=[True, True],inplace=True)
df_train.sort_values(by=['item_id','Z_WEEK_DATE'], ascending=[True, True],inplace=True)



## Creacion de la data total (training and testing) con identificador de ultima venta
##### union de dataset training y testing para poder hacer la misma procesamiento en los dos datasets

print('Release week')
# columna release significa la fecha en el cual se empezo a vender (demanda mayor que 0)
release_df = df_train[['item_id','date_block_num']][df_train[TARGET]>0].groupby(['item_id'])['date_block_num'].agg(['min']).reset_index()
release_df.columns = ['item_id','release']
print('1 df_auxiliar',df_auxiliar.shape)
df_auxiliar = merge_by_concat(df_auxiliar, release_df, ['item_id'])
print('2 df_auxiliar',df_auxiliar.shape)
# limpieza de memoria
del release_df

# las ventas que nunca se empezaron y tienen todo 0 en el release aparecen como NaN por ende lo mandamos a un valor por defecto 100
df_auxiliar['release'].fillna(100.0, inplace=True)

df_auxiliar['release'] = df_auxiliar['release'].astype(np.int16)
df_auxiliar = df_auxiliar.reset_index(drop=True)
print('3 df_auxiliar',df_auxiliar.shape)

# actualizamos nuestro release con respecto a la minima fecha de venta 
df_auxiliar['release'] = df_auxiliar['release'] - df_auxiliar['release'].min()
df_auxiliar['release'] = df_auxiliar['release'].astype(np.int16)




# guardamos el primer dataset referencial de toda la data para analizar training y testing con fecha de venta referencial
print('Save Part 1')
df_auxiliar.to_pickle(os.path.join(PATH_RESULTS,'dataset','grid_part_1.pkl'))
print('Size:', df_auxiliar.shape)


# Creacion de Features Temporales


# Definir nuestros parametros de trabajo con respecto a los puntos de corte temporales
END_TRAIN = df_auxiliar['date_block_num'].max()         # Last day in train set
print('END_TRAIN  :',END_TRAIN)

MAIN_INDEX = ['item_id','date_block_num']  # We can identify item by these columns
print('TARGET     :',TARGET)
print('END_TRAIN  :',END_TRAIN)
print('MAIN_INDEX :',MAIN_INDEX)
START_TRAIN = df_auxiliar['date_block_num'].min()         # First day in train set
print('START_TRAIN  :',START_TRAIN)

del df_auxiliar
gc.collect()


## Creacion de variables temporales del Z_WEEK_DATE (training and testing)
##### union de dataset training y testing para poder hacer la misma procesamiento en los dos datasets


# Creacion de features del dia de venta

# union de dataset training y testing para poder hacer la misma procesamiento en los dos datasets
df_train_aux = pd.read_pickle(os.path.join(PATH_RESULTS,'dataset','df_train.pkl'))
df_test_aux  = pd.read_pickle(os.path.join(PATH_RESULTS,'dataset','df_test.pkl'))    

df_auxiliar = pd.concat([df_train_aux,df_test_aux])
del df_train_aux
del df_test_aux
gc.collect()

df_auxiliar = df_auxiliar[['Z_WEEK_DATE', 'item_id', 'date_block_num']]
print('df_auxiliar',df_auxiliar.shape)

# Creacion de features del dia de venta
df_auxiliar             = fe_dates("Z_WEEK_DATE",df_auxiliar)
df_auxiliar['tm_wm']    = df_auxiliar['day'].apply(lambda x: ceil(x/7)).astype(np.int8) # 오늘 몇째주?
df_auxiliar['tm_w_end'] = (df_auxiliar['day_of_week']>=5).astype(np.int8)
df_auxiliar['tm_m_end'] = (df_auxiliar['tm_wm']>=3).astype(np.int8)
del df_auxiliar['Z_WEEK_DATE']
gc.collect()
# save features dates
print('Save part 2')

# Safe part 3
df_auxiliar.to_pickle(os.path.join(PATH_RESULTS,'dataset','grid_part_2.pkl'))
print('Size:', df_auxiliar.shape)

# We don't need calendar_df anymore
del df_auxiliar
gc.collect()


## Creacion de variables temporales del pasado (training and testing)

global grid_df

grid_df = pd.read_pickle(os.path.join(PATH_RESULTS,'dataset','grid_part_1.pkl'))
grid_df = grid_df.reset_index(drop=True)
grid_df.sort_values(by=['item_id','Z_WEEK_DATE'], ascending=[True, True],inplace=True)

# Lags ( demanda de los dias anteriores)
start_time = time.time()
print('Create lags')
LAG_DAYS = [col for col in range(SHIFT_DAY,SHIFT_DAY+WINDOW)]
print('len LAG_DAYS',len(LAG_DAYS))
grid_df = grid_df.assign(**{
        '{}_lag_shift_{}'.format(col, l): grid_df.groupby(['item_id'])[col].transform(lambda x: x.shift(l))
        for l in LAG_DAYS
        for col in [TARGET]
    })

grid_df.replace([np.inf, -np.inf,np.nan],0 , inplace=True)
print('grid_df',grid_df.shape)
grid_df.replace([np.inf, -np.inf, np.nan],0,inplace=True)
grid_df = grid_df.reset_index(drop=True)
gc.collect()
# Minify lag columns
for col in list(grid_df):
    if 'lag' in col:
        grid_df[col] = grid_df[col].astype(np.float16)

print('%0.2f min: Lags' % ((time.time() - start_time) / 60))





## Creacion de variables temporales del pasado tendencias de venta por modelo, puntop de venta , departamento y mas variables (training and testing)

### PROCESAMIENTO MULTITHREAD




grid_df[['Z_MODELO', 'Z_PUNTO_VENTA', 'Z_GAMA']] = grid_df['item_id'].str.split('|',expand=True)

# defino los grupos para hacer el analisis temporal
icols =  [  ['item_id'],
            ['Z_MODELO'],
            ['Z_PUNTO_VENTA'],
            ['Z_GAMA'],
            ['Z_MARCA'],
            ['Z_DEPARTAMENTO'],
    
            ['Z_MODELO', 'Z_PUNTO_VENTA'],
          
            ['Z_MODELO', 'Z_GAMA'],
            ['Z_MODELO', 'Z_MARCA'],
            ['Z_MODELO', 'Z_DEPARTAMENTO'],
    
            ['Z_PUNTO_VENTA', 'Z_GAMA'],
            ['Z_PUNTO_VENTA', 'Z_MARCA'],
            ['Z_PUNTO_VENTA', 'Z_DEPARTAMENTO'],
    
            ['Z_GAMA', 'Z_MARCA'],
            ['Z_GAMA', 'Z_DEPARTAMENTO'],
    
            ['Z_MARCA', 'Z_DEPARTAMENTO'],
        
            ]
# Rollings
start_time = time.time()
print('Create rolling aggs')
# Rollings
# with sliding shift


total_combinations = []
for d_shift in range(SHIFT_DAY,SHIFT_DAY+WINDOW+1): 
    print('Shifting period:', d_shift)
    for d_window in [2,4]:
        col_name = 'shift_'+str(d_shift)+'_roll_'+str(d_window)
        for group_columns in icols:
            for tipo in ['mean','std']:
                total_combinations.append([d_shift,d_window,group_columns,tipo])


def process_lags(x):
    global grid_df
    d_shift = x[0]
    d_window = x[1]
    group_columns = x[2]
    tipo = x[3]
    col_name = 'shift_'+str(d_shift)+'_roll_'+str(d_window)
    if tipo == 'mean':
        var = grid_df.groupby(group_columns)[TARGET].transform(lambda x: x.shift(d_shift).fillna(0).rolling(d_window,min_periods=1).mean()).astype(np.float16)
        return [col_name+'_mean_'+'_'.join(group_columns),var]
    if tipo == 'std':
        var = grid_df.groupby(group_columns)[TARGET].transform(lambda x: x.shift(d_shift).fillna(0).rolling(d_window,min_periods=1).std(ddof=0)).astype(np.float16)
        return [col_name+'_std_'+'_'.join(group_columns),var]
    if tipo == 'max':
        var = grid_df.groupby(group_columns)[TARGET].transform(lambda x: x.shift(d_shift).fillna(0).rolling(d_window,min_periods=1).max()).astype(np.float16)
        return [col_name+'_std_'+'_'.join(group_columns),var]


print('Inicio del procesamiento paralelo para crear features temporales por grupo')
results = Parallel(n_jobs=8, batch_size=64, backend="loky", verbose=0)(delayed(process_lags)(n) for n in tqdm(total_combinations))
n= len(results)

for i in tqdm(range(n)):
    va = results.pop()
    grid_df[va[0]] = va[1]    
    
print('Fin del procesamiento paralelo para crear features temporales por grupo')
print('OK')
print('%0.2f min: Lags' % ((time.time() - start_time) / 60))

del results
del grid_df['Z_MODELO']
del grid_df['Z_PUNTO_VENTA']
del grid_df['Z_GAMA']
del grid_df['Z_MARCA']
del grid_df['Z_DEPARTAMENTO']

gc.collect()



# guardar toda la data temporal
print('Save lags and rollings')
print(os.path.join(PATH_RESULTS,'dataset','lags_df_'+str(SHIFT_DAY)+'_completed.pkl'))
print('Size:', grid_df.shape)
grid_df.to_pickle(os.path.join(PATH_RESULTS,'dataset','lags_df_'+str(SHIFT_DAY)+'_completed.pkl'))


del grid_df
gc.collect()

## Creacion de variables temporales del pasado tendencias global de venta por modelo, punto de venta , departamento y mas variables de las 10 semanas anteriores a SEMANA_50 (training and testing)


grid_df = pd.read_pickle(os.path.join(PATH_RESULTS,'dataset','grid_part_1.pkl'))
grid_df[TARGET][grid_df['date_block_num']>(END_TRAIN-10)] = np.nan
base_cols = list(grid_df)

icols =  [
            ['Z_MODELO'],
            ['Z_PUNTO_VENTA'],
            ['Z_GAMA'],
            ['Z_MARCA'],
            ['Z_DEPARTAMENTO'],
    
            ['Z_MODELO', 'Z_PUNTO_VENTA'],
            ['Z_MODELO', 'Z_GAMA'],
            ['Z_MODELO', 'Z_MARCA'],
            ['Z_MODELO', 'Z_DEPARTAMENTO'],
    
            ['Z_PUNTO_VENTA', 'Z_GAMA'],
            ['Z_PUNTO_VENTA', 'Z_MARCA'],
            ['Z_PUNTO_VENTA', 'Z_DEPARTAMENTO'],
    
            ['Z_GAMA', 'Z_MARCA'],
            ['Z_GAMA', 'Z_DEPARTAMENTO'],
    
            ['Z_MARCA', 'Z_DEPARTAMENTO'],
        
            ['Z_MODELO', 'Z_PUNTO_VENTA', 'Z_GAMA'],
            ]

for col in icols:
    print('Encoding', col)
    col_name = '_'+'_'.join(col)+'_'
    grid_df['enc'+col_name+'mean'] = grid_df.groupby(col)[TARGET].transform('mean').astype(np.float32)
    grid_df['enc'+col_name+'std'] = grid_df.groupby(col)[TARGET].transform('std').astype(np.float32)
    grid_df['enc'+col_name+'max'] = grid_df.groupby(col)[TARGET].transform('max').astype(np.float32)
keep_cols = [col for col in list(grid_df) if col not in base_cols]

keep_cols = [col for col in list(grid_df) if col not in base_cols]
grid_df = grid_df[['Z_WEEK_DATE', 'item_id', 'date_block_num']+keep_cols]

# guardar la data global de tendencia
print('Save Mean/Std encoding')
print(os.path.join(PATH_RESULTS,'dataset','mean_encoding_df.pkl'))
grid_df.to_pickle(os.path.join(PATH_RESULTS,'dataset','mean_encoding_df.pkl'))

del grid_df
gc.collect()


# Archivos de variables temporales creadas

























