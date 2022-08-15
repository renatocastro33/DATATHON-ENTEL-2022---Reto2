import sys
sys.path.insert(1, '../../Src')
from utils.preprocessing import *
import training

import numpy as np
import pandas as pd
import os


from math import ceil
import warnings


warnings.filterwarnings('ignore')

PATH_DATASET = '../dataset/'
PATH_RESULTS = '../../results/Demanda/'


FILL = False
TARGET = 'Demanda'


N_FOLDS = 3
N_FEATURE_IMPORTANCE = 15

plt.rcParams['axes.facecolor']='white'
plt.rcParams['savefig.facecolor']='white'
SEED = 42
seed_everything(SEED)


warnings.simplefilter(action='ignore', category=FutureWarning)

# Lectura de data referencial para sacar metricas de analisis



submission_top_reference = 'Submission_73.csv'
print('reading .. ',os.path.join('../../results/',submission_top_reference))
result = pd.read_csv(os.path.join('../../results/',submission_top_reference))
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
result.head(1)



# Lectura de la data con las nuevas variables creadas



print('Reading dataset')
SHIFT=1
df_train  = pd.read_pickle(os.path.join(PATH_RESULTS,'dataset','df_train_fe_FILL_'+str(FILL)+'_SHIFT_'+str(SHIFT)+'.pkl'))
df_test   = pd.read_pickle(os.path.join(PATH_RESULTS,'dataset','df_test_fe_FILL_'+str(FILL)+'_SHIFT_'+str(SHIFT)+'.pkl'))

gc.collect()
print('df_train size :',df_train.shape)
print('df_test size  :',df_test.shape) 

features_names = set(df_train.columns)-set([TARGET,'date_block_num',TARGET+'_clipped','release','item_id',
                                            'Z_WEEK_DATE', 'Z_WEEK','date_block_num','year',
                                            'month', 'is_month_end', 'tm_m_end','is_month_start','tm_wm', 'tm_w_end','is_quarter_start','is_year_start',
                                             'day_of_week',
                                            'day_of_year', 'day',
                                           ]) 
print(len(features_names))
features_names = list(features_names)

df_train['week_of_month'] = df_train['week_of_month'].astype('category')
df_test['week_of_month']  = df_test['week_of_month'].astype('category')
total_features_names = features_names.copy()


# Filtro de datos por ventana de tiempo temporal [1 a 3 meses antes]


def get_feature_names_shift(shift,total_features_names):
    features_names = []

    for column in total_features_names:
        if 'shift' in column:
            number = column.split('shift_')[1]
            number = int(number.split('_')[0])
            if number>=shift and number<=(shift+2):
                features_names.append(column)
        else:
            features_names.append(column)
    return features_names


# Definir la ventana de testing para poder evaluar que el modelo predice bien el futuro
### esta ventana de testing es nuestra base de analisis en el training

TRAIN_START = '2021-08-01'
TEST_START = '2022-03-01'
TEST_END   = '2022-05-01'
print(df_train.shape)
df_train = df_train[(df_train['Z_WEEK_DATE'] >= TRAIN_START)]
print(df_train.shape)


# Entrenamiento con cross validation e importancia de datos





PATH_RESULTS = '../../results/Demanda/'
training.PATH_RESULTS = PATH_RESULTS
training.submission_analysis.TARGET  = TARGET


for SHIFT in range(1,3):
    print('*'*20)
    print('*'*20)
    print('WEEK = ',SHIFT)
    print('*'*20)
    print('*'*20)

    features_names = get_feature_names_shift(SHIFT,total_features_names)
    print('len base features = ',len(features_names))
    print('example base features =',features_names[:5])

    ################### CV MODEL RANDOM SPLIT ###############


    
    for idx,model_type in enumerate(['xgboost']):

    
        X_train      = df_train[((df_train['Z_WEEK_DATE'] >= TRAIN_START) &(df_train['Z_WEEK_DATE'] < TEST_START))&
                               (df_train['Z_WEEK_DATE'] != '2022-04-11')].copy() 
        X_test       = df_train[(df_train['Z_WEEK_DATE'] >= TEST_START) & (df_train['Z_WEEK_DATE'] < TEST_END)].copy() 

        print('*'*20)
        print('MODEL = ',model_type)
        print('CV MODEL RANDOM SPLIT model_type = ',model_type,'c_model_v2')
        print('*'*20)

        y_train = X_train[TARGET]
        y_test =  X_test[TARGET]
        y_submission = df_test[TARGET]
        X_train = X_train[features_names]
        X_test  = X_test[features_names]
        X_submission = df_test[features_names]        
        


        print('X_train total cv',X_train.shape)
        print('X_test          ',X_test.shape)
        print('X_submission    ',X_submission.shape)

        model_version = 'c_model_v2'
        training.TARGET = TARGET
        training.SHIFT = SHIFT

        df_submission,df_feature_importance = training.training_model_cv(model_type,model_version,X_train,y_train,
                                                            X_test,y_test,X_submission,df_test,result,N_FOLDS)

    
        
        
        #'''
        if idx ==0 :
            features_names = list(df_feature_importance['feature'][:15].values)
            
        ################### CV MODEL RANDOM SPLIT + FEATURE IMPORTANCE ###############
        important_features = list(df_feature_importance[:N_FEATURE_IMPORTANCE]['feature'])
        
        X_train      = df_train[(df_train['Z_WEEK_DATE'] >= TRAIN_START)&(df_train['Z_WEEK_DATE'] < TEST_START)].copy() 
        X_test       = df_train[(df_train['Z_WEEK_DATE'] >= TEST_START) & (df_train['Z_WEEK_DATE'] < TEST_END)].copy() 

        y_train = X_train[TARGET]
        y_test =  X_test[TARGET]
        y_submission = df_test[TARGET]

        X_train = X_train[important_features]
        X_test  = X_test[important_features]
        X_submission = df_test[important_features]

        print('X_train total cv',X_train.shape)
        print('X_test          ',X_test.shape)
        print('X_submission    ',X_submission.shape)

        print('*'*20)
        print('MODEL = ',model_type)
        print('CV MODEL RANDOM SPLIT + FEATURE IMPORTANCE model_type = ',model_type,'c_model_v3')
        print('*'*20)
        
        model_version = 'c_model_v3'
        training.TARGET = TARGET
        training.SHIFT = SHIFT


        df_submission,df_feature_importance = training.training_model_cv(model_type,model_version,X_train,y_train,
                                                               X_test,y_test,X_submission,df_test,result,N_FOLDS)
        #'''
        
        
        
        
