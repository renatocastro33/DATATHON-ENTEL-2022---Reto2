### Data Science Libraries ###
import pandas as pd # Librería de manipulación de data tabular
import numpy as np # Librería de operaciones matriciales
#import seaborn as sns # Librería de Data Visualization
#import matplotlib.pyplot as plt # Librería de Data Visualization
#import plotly.express as px # Librería avanzada de Data Visualizatión Dinámico
#from statsmodels.tsa.stattools import acf # Librería de Estadística

### Machine Learning Libraries ###
 
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.inspection import permutation_importance
#import catboost as cb
#import lightgbm as lgbm
#import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from prophet import Prophet

### OS and other Libraries ###
import inspect # Librería que permite listar variables del kernel presente
import os # Librería para realizar comandos de sistema operativo
import datetime # Librería para manejar clases de tiempo y fechas
import warnings # Librería para ignorar warnings
warnings.filterwarnings("ignore") #Ignoraremos cualquier warning con esta librería
import random # Librería para generar número(s) random
import pickle # Librería para leer archivos pickle
#import utils_entel # Funciones del archivo utils_entel.py que nos ayudará con nuestro preprocessing



train = pd.read_csv('../dataset/train/train_converted.csv') # Lectura del output train del notebook Preprocessing
test = pd.read_csv('../dataset/test/test_converted.csv') # Lectura del output test del notebook Preprocessing


descriptive_columns = ['Z_MARCA', 'Z_GAMA', 'Z_MODELO', 'Z_DEPARTAMENTO', 'Z_PUNTO_VENTA'] # Tener en cuenta nuestras variables descriptivas

train['ID'] = train['Z_MODELO'] + '|' + train['Z_PUNTO_VENTA'] + '|' + train['Z_GAMA']
test['ID'] = test['Z_MODELO'] + '|' + test['Z_PUNTO_VENTA'] + '|' + test['Z_GAMA']

m = Prophet(interval_width=0.95,weekly_seasonality=False , daily_seasonality=True, yearly_seasonality=True)
m.fit(train[['Z_WEEK_DATE', 'Demanda']].groupby('Z_WEEK_DATE').sum().reset_index().rename(columns = {'Z_WEEK_DATE':'ds', 'Demanda':'y'}))
future = m.make_future_dataframe(periods=10, freq = 'W')
fcst = m.predict(future)

print(fcst)

### Lo que hacemos a continuación es un baseline que gracias al entendimiento de variables importantes del catboost se implementará.
### Buscamos el ShareOfMarket de cada ID con respecto a la Demanda Nacional (DemandaID/DemandaNacional) por semana.
### A este ShareOfMarket semanal le llamaremos el nombre de "ratio"
### A continuación sacaremos el ratio prom de las ultimas 3 semanas para la última semana del periodo de análisis y se le aplicará con respecto a la Demanda Nacional Total


base = train[['ID', 'Demanda', 'Z_WEEK_DATE']].groupby(['ID', 'Z_WEEK_DATE']).sum().sort_values('Demanda' , ascending = [False]).reset_index()
base_cum = train[['Demanda', 'Z_WEEK_DATE']].groupby([ 'Z_WEEK_DATE']).sum().sort_values('Demanda' , ascending = [False]).reset_index()

pred = fcst[['ds', 'yhat']]
pred.columns = ['Z_WEEK_DATE', 'yhat']

base = base.merge(base_cum[['Z_WEEK_DATE', 'Demanda']].rename(columns = {'Demanda': 'Demanda_Total'}), on = 'Z_WEEK_DATE', how = "left")
base['ratio'] = base['Demanda']*100/base['Demanda_Total']

aux = base[base['Z_WEEK_DATE'] >= '2022-04-11'][['ID', 'ratio']].groupby('ID').mean().reset_index()

aux.sort_values('ratio', ascending = [False])


test = test.merge(aux, on='ID', how = "left") ### Hacemos merge a los ratios de los ID con el test.
pred_train = pred[pred['Z_WEEK_DATE'] < '2022-05-01']
pred_train['Z_WEEK_DATE'] = pred_train['Z_WEEK_DATE'].astype(str)


train = train.merge(pred_train, on = 'Z_WEEK_DATE', how = "left")
train['yhat_fixed'] = train['yhat']

pred_test = pred[pred['Z_WEEK_DATE'] >= '2022-05-01']
pred_test['Z_WEEK_DATE'] = np.array(['2022-05-02', '2022-05-09', '2022-05-16', '2022-05-23', 
                                '2022-05-30', '2022-06-06', '2022-06-13', '2022-06-20',
                                '2022-06-27', '2022-07-04']) 

test = test.merge(pred_test, on = 'Z_WEEK_DATE', how = "left")

test.loc[:,'Demanda'] = test['ratio']*test['yhat']/100

### Así mismo aplicaremos penalidades cada 3 semanas que pasen. Debido a la coyuntura política nuestro forecasting inicial buscaba un repunte en julio, sin embargo 
### por temas económicos y políticos que vimos quizás fue afectado. Entonces aquí se puede manejar varias formar de aplicar escenarios optimistas, neutros o pesimistas de
### acuerdo a la coyuntura mundiall. Estas penalidades ayudaron con el pronóstico.

test.loc[test['Z_WEEK_DATE'] == '2022-05-02', 'Demanda'] = test[test['Z_WEEK_DATE'] == '2022-05-02']['Demanda']*0.87 # Tope 0.87
test.loc[test['Z_WEEK_DATE'] == '2022-05-09', 'Demanda'] = test[test['Z_WEEK_DATE'] == '2022-05-09']['Demanda']*0.87 # Tope 0.87
test.loc[test['Z_WEEK_DATE'] == '2022-05-16', 'Demanda'] = test[test['Z_WEEK_DATE'] == '2022-05-16']['Demanda']*0.73 # Tope 0.73

test.loc[test['Z_WEEK_DATE'] == '2022-05-23', 'Demanda'] = test[test['Z_WEEK_DATE'] == '2022-05-23']['Demanda']*0.65 # Tope 0.65
test.loc[test['Z_WEEK_DATE'] == '2022-05-30', 'Demanda'] = test[test['Z_WEEK_DATE'] == '2022-05-30']['Demanda']*0.54 # Tope 0.55 posible 0.53
test.loc[test['Z_WEEK_DATE'] == '2022-06-06', 'Demanda'] = test[test['Z_WEEK_DATE'] == '2022-06-06']['Demanda']*0.62 # Top 0.62 posible 0.64 
test.loc[test['Z_WEEK_DATE'] == '2022-06-13', 'Demanda'] = test[test['Z_WEEK_DATE'] == '2022-06-13']['Demanda']*0.64 # Tope 0.64

test.loc[test['Z_WEEK_DATE'] == '2022-06-20', 'Demanda'] = test[test['Z_WEEK_DATE'] == '2022-06-20']['Demanda']*0.67 # Tope 0.67
test.loc[test['Z_WEEK_DATE'] == '2022-06-27', 'Demanda'] = test[test['Z_WEEK_DATE'] == '2022-06-27']['Demanda']*0.52 # Tope 0.52
test.loc[test['Z_WEEK_DATE'] == '2022-07-04', 'Demanda'] = test[test['Z_WEEK_DATE'] == '2022-07-04']['Demanda']*0.40 # Tope 0.40

test.loc[test['Z_WEEK_DATE'] == '2022-05-02', 'yhat_fixed'] = test[test['Z_WEEK_DATE'] == '2022-05-02']['yhat']*0.87 # Tope 0.87
test.loc[test['Z_WEEK_DATE'] == '2022-05-09', 'yhat_fixed'] = test[test['Z_WEEK_DATE'] == '2022-05-09']['yhat']*0.87 # Tope 0.87
test.loc[test['Z_WEEK_DATE'] == '2022-05-16', 'yhat_fixed'] = test[test['Z_WEEK_DATE'] == '2022-05-16']['yhat']*0.73 # Tope 0.73

test.loc[test['Z_WEEK_DATE'] == '2022-05-23', 'yhat_fixed'] = test[test['Z_WEEK_DATE'] == '2022-05-23']['yhat']*0.65 # 0.62
test.loc[test['Z_WEEK_DATE'] == '2022-05-30', 'yhat_fixed'] = test[test['Z_WEEK_DATE'] == '2022-05-30']['yhat']*0.54
test.loc[test['Z_WEEK_DATE'] == '2022-06-06', 'yhat_fixed'] = test[test['Z_WEEK_DATE'] == '2022-06-06']['yhat']*0.62
test.loc[test['Z_WEEK_DATE'] == '2022-06-13', 'yhat_fixed'] = test[test['Z_WEEK_DATE'] == '2022-06-13']['yhat']*0.64

test.loc[test['Z_WEEK_DATE'] == '2022-06-20', 'yhat_fixed'] = test[test['Z_WEEK_DATE'] == '2022-06-20']['yhat']*0.67 # Tope 0.67
test.loc[test['Z_WEEK_DATE'] == '2022-06-27', 'yhat_fixed'] = test[test['Z_WEEK_DATE'] == '2022-06-27']['yhat']*0.52 # Tope 0.52
test.loc[test['Z_WEEK_DATE'] == '2022-07-04', 'yhat_fixed'] = test[test['Z_WEEK_DATE'] == '2022-07-04']['yhat']*0.40 # Tope 0.40

test['Demanda_tuned'] = test['Demanda']
test['Demanda_prophet'] = test['ratio']*test['yhat']/100 


### Si bien es cierto los ratios promedio ult 3 semanas funcionó muy bien para posicionarnos con 2.04 en el public leaderboard. Se buscó ensembles 
### con otros ratios, el que nos funcionó bastante bien y tiene mucho sentido es el ratio de la última semana. 
### Realizar un ensemble con estos ratios*Demanda Proyectada Prophet y ratios_avg_ult_3_sem*Demanda Proyectada Prophet funcionó muy bien.

aux_last_trx_app = base[base['Z_WEEK_DATE'] == '2022-04-25'][['ID', 'ratio']].groupby('ID').mean().reset_index()
aux_last_trx_app.rename(columns = {'ratio':'ratio_last_trx_app'}, inplace = True)
aux_last_trx_app.sort_values('ratio_last_trx_app', ascending = [False])


test_ratio_merged = test.copy()

test_ratio_merged = test_ratio_merged.merge(aux_last_trx_app, on='ID', how = "left")

test_ratio_merged.loc[:,'Demanda_last_trx_app'] = test_ratio_merged['ratio_last_trx_app']*test['yhat']/100

test_ratio_merged.loc[test_ratio_merged['Z_WEEK_DATE'] == '2022-05-02', 'Demanda_last_trx_app'] = test_ratio_merged[test_ratio_merged['Z_WEEK_DATE'] == '2022-05-02']['Demanda_last_trx_app']*0.87 # Tope 0.87
test_ratio_merged.loc[test_ratio_merged['Z_WEEK_DATE'] == '2022-05-09', 'Demanda_last_trx_app'] = test_ratio_merged[test_ratio_merged['Z_WEEK_DATE'] == '2022-05-09']['Demanda_last_trx_app']*0.87 # Tope 0.87
test_ratio_merged.loc[test_ratio_merged['Z_WEEK_DATE'] == '2022-05-16', 'Demanda_last_trx_app'] = test_ratio_merged[test_ratio_merged['Z_WEEK_DATE'] == '2022-05-16']['Demanda_last_trx_app']*0.73 # Tope 0.73

test_ratio_merged.loc[test_ratio_merged['Z_WEEK_DATE'] == '2022-05-23', 'Demanda_last_trx_app'] = test_ratio_merged[test_ratio_merged['Z_WEEK_DATE'] == '2022-05-23']['Demanda_last_trx_app']*0.65 # Tope 0.65
test_ratio_merged.loc[test_ratio_merged['Z_WEEK_DATE'] == '2022-05-30', 'Demanda_last_trx_app'] = test_ratio_merged[test_ratio_merged['Z_WEEK_DATE'] == '2022-05-30']['Demanda_last_trx_app']*0.54 # Tope 0.55 posible 0.53
test_ratio_merged.loc[test_ratio_merged['Z_WEEK_DATE'] == '2022-06-06', 'Demanda_last_trx_app'] = test_ratio_merged[test_ratio_merged['Z_WEEK_DATE'] == '2022-06-06']['Demanda_last_trx_app']*0.62 # Top 0.62 posible 0.64 
test_ratio_merged.loc[test_ratio_merged['Z_WEEK_DATE'] == '2022-06-13', 'Demanda_last_trx_app'] = test_ratio_merged[test_ratio_merged['Z_WEEK_DATE'] == '2022-06-13']['Demanda_last_trx_app']*0.64 # Tope 0.64

test_ratio_merged.loc[test_ratio_merged['Z_WEEK_DATE'] == '2022-06-20', 'Demanda_last_trx_app'] = test_ratio_merged[test_ratio_merged['Z_WEEK_DATE'] == '2022-06-20']['Demanda_last_trx_app']*0.67 # Tope 0.67
test_ratio_merged.loc[test_ratio_merged['Z_WEEK_DATE'] == '2022-06-27', 'Demanda_last_trx_app'] = test_ratio_merged[test_ratio_merged['Z_WEEK_DATE'] == '2022-06-27']['Demanda_last_trx_app']*0.52 # Tope 0.52
test_ratio_merged.loc[test_ratio_merged['Z_WEEK_DATE'] == '2022-07-04', 'Demanda_last_trx_app'] = test_ratio_merged[test_ratio_merged['Z_WEEK_DATE'] == '2022-07-04']['Demanda_last_trx_app']*0.40 # Tope 0.40


### También ayudó los ratios de la penúltima semana. No con la misma intensidad de la última pero si permite suavizar algún efecto de canibalización y cambios en el ShareOfMarket de 
### un ID.

aux_pult_trx = base[base['Z_WEEK_DATE'] == '2022-04-18'].sort_values('ratio', ascending = [False])[['ID', 'ratio']]
aux_pult_trx.rename(columns = {'ratio': 'ratio_pult_trx'}, inplace = True)

test_ratio_merged = test_ratio_merged.merge(aux_pult_trx, on='ID', how = "left")

test_ratio_merged.loc[:,'Demanda_pult_trx'] = test_ratio_merged['ratio_pult_trx']*test['yhat']/100

test_ratio_merged.loc[test_ratio_merged['Z_WEEK_DATE'] == '2022-05-02', 'Demanda_pult_trx'] = test_ratio_merged[test_ratio_merged['Z_WEEK_DATE'] == '2022-05-02']['Demanda_pult_trx']*0.87 # Tope 0.87
test_ratio_merged.loc[test_ratio_merged['Z_WEEK_DATE'] == '2022-05-09', 'Demanda_pult_trx'] = test_ratio_merged[test_ratio_merged['Z_WEEK_DATE'] == '2022-05-09']['Demanda_pult_trx']*0.87 # Tope 0.87
test_ratio_merged.loc[test_ratio_merged['Z_WEEK_DATE'] == '2022-05-16', 'Demanda_pult_trx'] = test_ratio_merged[test_ratio_merged['Z_WEEK_DATE'] == '2022-05-16']['Demanda_pult_trx']*0.73 # Tope 0.73

test_ratio_merged.loc[test_ratio_merged['Z_WEEK_DATE'] == '2022-05-23', 'Demanda_pult_trx'] = test_ratio_merged[test_ratio_merged['Z_WEEK_DATE'] == '2022-05-23']['Demanda_pult_trx']*0.65 # Tope 0.65
test_ratio_merged.loc[test_ratio_merged['Z_WEEK_DATE'] == '2022-05-30', 'Demanda_pult_trx'] = test_ratio_merged[test_ratio_merged['Z_WEEK_DATE'] == '2022-05-30']['Demanda_pult_trx']*0.54 # Tope 0.55 posible 0.53
test_ratio_merged.loc[test_ratio_merged['Z_WEEK_DATE'] == '2022-06-06', 'Demanda_pult_trx'] = test_ratio_merged[test_ratio_merged['Z_WEEK_DATE'] == '2022-06-06']['Demanda_pult_trx']*0.62 # Top 0.62 posible 0.64 
test_ratio_merged.loc[test_ratio_merged['Z_WEEK_DATE'] == '2022-06-13', 'Demanda_pult_trx'] = test_ratio_merged[test_ratio_merged['Z_WEEK_DATE'] == '2022-06-13']['Demanda_pult_trx']*0.64 # Tope 0.64

test_ratio_merged.loc[test_ratio_merged['Z_WEEK_DATE'] == '2022-06-20', 'Demanda_pult_trx'] = test_ratio_merged[test_ratio_merged['Z_WEEK_DATE'] == '2022-06-20']['Demanda_pult_trx']*0.67 # Tope 0.67
test_ratio_merged.loc[test_ratio_merged['Z_WEEK_DATE'] == '2022-06-27', 'Demanda_pult_trx'] = test_ratio_merged[test_ratio_merged['Z_WEEK_DATE'] == '2022-06-27']['Demanda_pult_trx']*0.52 # Tope 0.52
test_ratio_merged.loc[test_ratio_merged['Z_WEEK_DATE'] == '2022-07-04', 'Demanda_pult_trx'] = test_ratio_merged[test_ratio_merged['Z_WEEK_DATE'] == '2022-07-04']['Demanda_pult_trx']*0.40 # Tope 0.40


test_ratio_merged['Demanda_65_35'] = test_ratio_merged['Demanda']*0.65 + 0.35*test_ratio_merged['Demanda_last_trx_app']

test_ratio_merged['Demanda_65_35__085_015'] = test_ratio_merged['Demanda_65_35']*0.85 + 0.15*test_ratio_merged['Demanda_pult_trx'] #Este ensemble funcionó muy bien.

test_ratio_merged['Demanda'] = test_ratio_merged['Demanda_65_35__085_015'] 
# Se lo pasamos al valor de la demanda. Si bien es cierto ya se logró realizar un buen resultado
# con posiblemente realizando un buen forecasting de la demanda nacional, pero falta mejorar por ID.
# El resultado de estos ensembles será usado como Stacking en un XGBoost.

submit = test_ratio_merged.copy()
reverse_mapping_file = '../utils/reverse_dict_mapping_list.txt'

with open(reverse_mapping_file, 'rb') as f:
    reverse_mapping = pickle.load( f)

i=0
for column in descriptive_columns:
    submit[column] = submit[column].map(reverse_mapping[i])
    i+=1
    
submit['ID'] = submit['Z_MODELO'] + '|' + submit['Z_PUNTO_VENTA'] + '|' + submit['Z_GAMA'] + '|' + submit['Z_WEEK']
submission = submit[['ID','Demanda']].groupby('ID').sum().reset_index()
submission[['ID', 'Demanda']].to_csv('../Results/Output_Prophet.csv', index = False, sep = ',')