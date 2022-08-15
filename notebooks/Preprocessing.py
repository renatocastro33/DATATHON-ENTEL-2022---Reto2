### Librerías de Ciencia de Datos ###
import numpy as np # Librería de operaciones matriciales
import pandas as pd # Librería de manipulación de data tabular
#import seaborn as sns # Librería de Data Visualization

### Otras Librerías Útiles ###
import inspect # Librería que permite listar variables del kernel presente
import os # Librería para realizar comandos de sistema operativo
import datetime # Librería para manejar clases de tiempo y fechas
import warnings # Librería para ignorar warnings
warnings.filterwarnings('ignore') #Ignoraremos cualquier warning con esta librería
import random # Librería para generar número(s) random
import pickle # Librería para leer archivos pickle
import utils_entel # Funciones del archivo utils_entel.py que nos ayudará con nuestro preprocessing

descriptive_columns = ['Z_MARCA', 'Z_GAMA', 'Z_MODELO',
                       'Z_DEPARTAMENTO', 'Z_PUNTO_VENTA']


directory_path = os.getcwd() # Guardamos el directorio donde nos encontramos para evitar hardcodeo.
raw_train = pd.read_csv(os.path.join(directory_path, '..\\dataset\\train\\train.csv')) # Usando el esquema descrito arriba procedemos a el archivo de train
raw_test = pd.read_csv(os.path.join(directory_path, '..\\dataset\\test\\test.csv')) # Usando el esquema descrito arriba procedemos a el archivo de test


### Hallamos la cantidad de registros que hay en cada uno ###
utils_entel.decorative_function_dataframe_value('Shape value of ',utils_entel.get_variable_name(raw_train), raw_train.shape)
utils_entel.decorative_function_dataframe_value('Shape value of ',utils_entel.get_variable_name(raw_test), raw_test.shape)


### MAPEAR VALORES ANÓNIMOS EN MÁS SIMPLES ###
'''
Usamos la función mapping_encoded_columns del módulo utils_entel.py (revisar el file para entender lo que hace la función a detalle).
El objetivo es tener valores más simple por las columnas descriptivas las cuales son la marca, modelo, etc.
'''
(raw_train, mapping_list) = utils_entel.mapping_encoded_columns(raw_train, descriptive_columns)
raw_test = utils_entel.mapping_encoded_columns(raw_test, descriptive_columns, mapping_list)


### Realizamos una validación de valores para ver si hay misma igualdad en registros de cada columna entre ambos dataframes
### Es decir si tengo 11,888 registros en el raw_train la marca MAR_1      
### entonces debo tener esa misma cantidad en el raw_test
[ utils_entel.columns_info(raw_train, raw_test, \
               column) for column in descriptive_columns]
### Conclusion: Diferencia en validación es un conjunto vacío set(). Por lo que contiene los misma cantidad de valores que el original, transformación exitosa.###


### MELT TABLE ###
'''
Se realiza la operación de mel con respecto a las semanas para tenerlas en 1 sola columna sus valores.
'''
week_cols_train = sorted(list(set(raw_train.columns)-set(descriptive_columns)))
week_cols_test = sorted(list(set(raw_test.columns)-set(descriptive_columns)))

train = pd.melt(raw_train, 
                    id_vars=descriptive_columns,
                    value_vars=week_cols_train,
                    var_name='Z_WEEK',
                    value_name='Demanda')

test = pd.melt(raw_test, 
                    id_vars=descriptive_columns,
                    value_vars=week_cols_test,
                    var_name='Z_WEEK',
                    value_name='Demanda')

test['Demanda'] = np.nan ### Reemplazar los valores 99 con nan. Buena práctica para no confundirnos después.


### Creamos la columna Z_WEEK_DATE que tiene el primer día de la semana en formato fecha real para el raw_train
### Para entender la función detalladamente leer el módulo utils_entel.py
train = utils_entel.week_mapping(train, 'Z_WEEK', datetime.date(2021, 5, 17))

### Creamos la columna Z_WEEK_DATE que tiene el primer día de la semana en formato fecha real para el raw_test
### Para entender la función detalladamente leer el módulo utils_entel.py
test = utils_entel.week_mapping(test, 'Z_WEEK', datetime.date(2022, 5, 2)) 


### Extraemos el día, mes, y año para el forecasting y el EDA ###
train['Z_DAY'] = train['Z_WEEK_DATE'].map(lambda x: x.day)
train['Z_MONTH'] = train['Z_WEEK_DATE'].map(lambda x: x.month)
train['Z_YEAR'] = train['Z_WEEK_DATE'].map(lambda x: x.year)

test['Z_DAY'] = test['Z_WEEK_DATE'].map(lambda x: x.day)
test['Z_MONTH'] = test['Z_WEEK_DATE'].map(lambda x: x.month)
test['Z_YEAR'] = test['Z_WEEK_DATE'].map(lambda x: x.year)


train.to_csv('..\\dataset\\train\\train_converted.csv', sep = ',', index = False) ### Guardamos el nuevo dataset de raw_train para propositos de modelo.
test.to_csv('..\\dataset\\test\\test_converted.csv', sep = ',', index = False) ### Guardamos el nuevo dataset de raw_test para propositos de modelo.


### Revertimos el mapeo del diccionario mapping_list con la finalidad de regresar los valores originales anonimizados.
### ORIGINAL ENCODED VALUES THE DESCRIPTIVE COLUMNS###
reverse_mapping = [{v: k for k, v in dictionary.items()} for dictionary in mapping_list ]


reverse_mapping_file = '..\\utils\\reverse_dict_mapping_list.txt' ### Guardamos el diccionario en un archivo txt que usaremos para mandar los submissions a Kaggle.

with open(reverse_mapping_file, 'wb') as f:
    pickle.dump(reverse_mapping, f)
    
    
with open(reverse_mapping_file, 'rb') as f:
    reverse_mapping_read = pickle.load( f)
    
    
assert reverse_mapping_read == reverse_mapping, 'ERROR, THEY ARE NOT THE SAME DICTIONARY' # Nos aseguramos que sean el mismo diccionario con el del archivo txt 