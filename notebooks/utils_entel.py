### Librerías de Ciencia de Datos ###
import numpy as np # Librería de operaciones matriciales
import pandas as pd # Librería de manipulación de data tabular
import seaborn as sns # Librería de Data Visualization

### Otras Librerías Útiles ###
import inspect # Librería que permite listar variables del kernel presente
import os # Librería para realizar comandos de sistema operativo
import datetime # Librería para manejar clases de tiempo y fechas
import warnings # Librería para ignorar warnings
warnings.filterwarnings('ignore') #Ignoraremos cualquier warning con esta librería
import random # Librería para generar número(s) random
import pickle # Librería para leer archivos pickle



### Funciones Útiles para el Preprocessing ###

def get_variable_name(variable):
    """Nos devuelve el nombre definido de la variable en el kernel de Python.

    Args:
        variable: object. Objeto cuyo nombre se desea obtener

    Output:
        variable_name: str. Nombre definido del objeto
    """
    for fi in reversed(inspect.stack()):
        names = [var_name for var_name, var_val in fi.frame.f_locals.items() if var_val is variable]
        if len(names) > 0:
            return names[0]
            
def decorative_function_dataframe_value(text, df_name, method):
    """Función decorativa que imprime valores del dataframe
    
    Args:
        text: str. Texto para propositos decorativos
        df: str. Nombre del Dataframe
        method: object. Objeto a ser impreso. (e.g. array, lists, str, y otros)
    Output:
        
    """
    print(' '.join([text, df_name,  ' is: ', str(method)]))

    
def columns_info(df1, df2, column ):
    '''Describe el conteo de valores y busca las diferencias entre ambos dataframes inputs.
    
    Args:
        df1: object. Dataframe 1 que contiene la columna a ser descrita.
        df2: object. Dataframe 2 la cual se comparará con el Dataframe 1.
        column: object. Columna a ser descrita
    Outputs:
        Ninguno
    '''
    print('###### ', column , ' TRAIN INFO ####\n')
    print(df1[column].value_counts())
    
    print('###### ', column , ' TEST INFO ####\n')
    print(df2[column].value_counts())
    
    print('The set with the columns that have different values: ', 
          set(df1[column].values) - \
          set(df2[column].values))
    print('\n##############################\n')
    
    
def mapping_encoded_columns(df, columns, mapping_list = None):
    '''Mapea las columnas anonimizadas a valores más simple 
    con la finalidad de Exploratory Data Analysis. El valor que regresa es de acuerdo al
    index del ranking del metodo value_counts().
    
    Args:
        df: dataframe. Dataframe que tiene la columna mapeada.
        column: list. Nombre de columnas a mapear.
    
    Optional:
        mapping_list: list. Un args opcional es si ya existe el Mapping_list, esto se usará para el raw_test ya que se aplicará el mismo map del raw_train
        
    Outputs:
        new_df: dataframe. Dataframe con las columnas mapeadas en valores simples.
    
    '''
    new_df = df.copy()
    
    if mapping_list == None:
        mapping_list = []
        for column in columns:
            new_dict = dict()
            index = 1
            for (value, count) in dict(new_df[column].value_counts()).items():
                new_dict[value] = column.replace('Z_', '')[:3].replace('PUN', 'PVENT') + \
                                                            '_' + str(index)
                index+=1

            new_df[column] = new_df[column].map(new_dict)
            mapping_list.append(new_dict)
    
        return (new_df, mapping_list)
    else:
        ind = 1
        for column in columns:
            new_df[column] = new_df[column].map(mapping_list[ind-1])
            ind += 1
    
        return new_df
    
def week_mapping(df, week_column, start_week_date):
    '''Create a mapped date variable from the week str values, using a start_week_date as inital point. Creamos un diccionario mapping que transforme los valores de las semanas 'SEMANA_01' a 
    una respectiva fecha real, en este caso usaremos el primer día de la semana relacionada.
    e.g. 'SEMANA_01' -> datetime.date(2022,5,17), donde la SEMANA_01 comprende del 2021-05-17 a 2021-05-23
    
    Args:
            df: 'dataframe'. Dataframe el cual será aplicado el week_mapping
        week_column: 'str'. Nombre de la columna a mapearse.
        start_week_date: 'datetime.date'. Primer día de la semana en type datetime.date
    Outputs:
        aux_df: 'dataframe'. Dataframe con la variable de semana transformada a su primer día real.
    '''
    assert type(start_week_date) == datetime.date, 'start_week_date is not a datetime.date type variable'
    
    aux_df = df.copy()
    week_mapping = dict()
    week_date = start_week_date
    for week_str in sorted(aux_df[week_column].unique()):
        week_mapping[week_str] = week_date
        week_date += datetime.timedelta(days = 7)
    
    date_week_column = week_column + '_DATE'
    aux_df[date_week_column] = aux_df[week_column].map(week_mapping)
    
    return aux_df