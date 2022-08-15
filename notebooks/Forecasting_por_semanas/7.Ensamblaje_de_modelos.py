import sys
sys.path.insert(1, '../../Src')
from utils.preprocessing import *
import training

import numpy as np
import pandas as pd
import os

import glob
from math import ceil
from sklearn.metrics import mean_squared_error


warnings.filterwarnings('ignore')

PATH_DATASET = '../../dataset/'
PATH_RESULTS = '../../results/Demanda_ventana_custom/'


FILL = False
TARGET = 'Demanda'


N_FOLDS = 3
N_FEATURE_IMPORTANCE = 15

plt.rcParams['axes.facecolor']='white'
plt.rcParams['savefig.facecolor']='white'
SEED = 42
seed_everything(SEED)
training.PATH_RESULTS = PATH_RESULTS
training.submission_analysis.TARGET  = TARGET

def get_result(name,path_results):
    print('reading .. ',os.path.join('../../results/',name))
    result = pd.read_csv(os.path.join('../../results/',name))
    descriptive_columns = ['Z_MODELO','Z_PUNTO_VENTA','Z_GAMA','Z_WEEK']
    reverse_mapping_file = '../../utils/reverse_dict_mapping_list.txt'

    result[['Z_MODELO','Z_PUNTO_VENTA','Z_GAMA','Z_WEEK']] = result['ID'].str.split('|',expand=True)

    with open(reverse_mapping_file, 'rb') as f:
        reverse_mapping = pickle.load( f)

    descriptive_columns = ['Z_MARCA', 'Z_GAMA', 'Z_MODELO',
                           'Z_DEPARTAMENTO', 'Z_PUNTO_VENTA']
    i=0
    for column in descriptive_columns:
        if column in ['Z_MODELO','Z_PUNTO_VENTA','Z_GAMA','Z_WEEK']:
            inv_reverse_mapping = {v: k for k, v in reverse_mapping[i].items()}

            result[column] = result[column].map(inv_reverse_mapping)
        i+=1

    result = result.rename(columns = {TARGET: TARGET+'_real'})
    
    result['Demanda_pred'] = 0

    WEEKS_LIST = {}

    TOTAL_FILES = 0
    for week in range(1,11):
        for idx,model_type in enumerate(['xgboost']):
            for model in ['c_model_v3','c_model_v2']:
                path_folder = os.path.join(path_results,model_type,model,'shift_'+str(week))
                list_filename_csv = glob.glob(os.path.join(path_folder,"*_submission.csv"))
                if len(list_filename_csv)>0:
                    score = float(list_filename_csv[0].split('rmse_cv_test_')[-1].split('_')[0])

                    if score>(2.15+week*50):#*0.15):
                        break
                    print('reading ..',list_filename_csv[0])
                    TOTAL_FILES+=1
                    filename_csv = list_filename_csv[0]
                    pred = pd.read_csv(filename_csv).rename(columns={"Demanda":"Demanda_pred"}).reset_index(drop=True)
                    SEMANA = 'SEMANA_'+str(50+week)

                    if SEMANA in WEEKS_LIST.keys():
                        WEEKS_LIST[SEMANA]+=1.0
                    else:
                        WEEKS_LIST[SEMANA]=1.0

                    sub_group = result[["Z_MODELO","Z_PUNTO_VENTA","Z_GAMA","Z_WEEK"]][result['Z_WEEK']==SEMANA].merge(
                        pred[["Z_MODELO","Z_PUNTO_VENTA","Z_GAMA","Z_WEEK","Demanda_pred"]],how='left').reset_index(drop=True)

                    result['Demanda_pred'][result['Z_WEEK']==SEMANA] += sub_group['Demanda_pred'].values
    print('TOTAL_FILES = ',TOTAL_FILES)

    for SEMANA in WEEKS_LIST:
        result['Demanda_pred'][result['Z_WEEK']==SEMANA] /= WEEKS_LIST[SEMANA]
    result['Demanda_pred'][~result['Z_WEEK'].isin(WEEKS_LIST.keys())]  = result['Demanda_real'][~result['Z_WEEK'].isin(WEEKS_LIST.keys())] 


    return result


name = 'Submission_73.csv'
path_results = '../../results/Demanda/'
result_model1 = get_result(name,path_results)
result_model1['Demanda_pred'][result_model1['Z_WEEK']=='SEMANA_52']  = 0.2*result_model1['Demanda_pred'][result_model1['Z_WEEK']=='SEMANA_52'] + 0.8*result_model1['Demanda_real'][result_model1['Z_WEEK']=='SEMANA_52']

rmse_target = mean_squared_error(result_model1[TARGET+'_pred'],result_model1[TARGET+'_real'], squared=False)

submissions_path = os.path.join('../../results/','submission_model1.csv')
print('saving submission ..',submissions_path)

print('Final score referencial mean_squared_error')
print('RMSE Score referencial TARGET:',rmse_target)

if True:
    print('saving ..')
    result_model1[['ID','Demanda_pred']].rename(columns={"Demanda_pred":"Demanda"}).to_csv(submissions_path,index=None)
    print('saved ! ')
    
    
    
name = 'submission_model1.csv'
path_results = '../../results/Demanda_ventana_custom/'
result_model2 = get_result(name,path_results)
result_model2['Demanda_pred'][result_model2['Z_WEEK']=='SEMANA_51']  = 0*result_model2['Demanda_pred'][result_model2['Z_WEEK']=='SEMANA_51'] + 1*result_model2['Demanda_real'][result_model2['Z_WEEK']=='SEMANA_51']
result_model2['Demanda_pred'][result_model2['Z_WEEK']=='SEMANA_52']  = 0*result_model2['Demanda_pred'][result_model2['Z_WEEK']=='SEMANA_52'] + 1*result_model2['Demanda_real'][result_model2['Z_WEEK']=='SEMANA_52']
result_model2['Demanda_pred'][result_model2['Z_WEEK']=='SEMANA_53']  = 0.5*result_model2['Demanda_pred'][result_model2['Z_WEEK']=='SEMANA_53'] + 0.5*result_model2['Demanda_real'][result_model2['Z_WEEK']=='SEMANA_53']
casteo_semanas =['SEMANA_51','SEMANA_52','SEMANA_53']
result_model2['Demanda_pred'][~result_model2['Z_WEEK'].isin(casteo_semanas)]  = result_model2['Demanda_real'][~result_model2['Z_WEEK'].isin(casteo_semanas)] 
    

submissions_path = os.path.join('../../results/','submission_model_final.csv')
print('saving submission ..',submissions_path)

if True:
    print('saving ..')
    result_model2[['ID','Demanda_pred']].rename(columns={"Demanda_pred":"Demanda"}).to_csv(submissions_path,index=None)
    print('saved ! ')