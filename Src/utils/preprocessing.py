# General imports
import os
import numpy as np
import os, sys, gc, time, warnings, pickle, psutil, random
from math import ceil
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
from IPython.display import display, HTML
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
import gc


sns.set_theme(style="ticks", color_codes=True)
warnings.filterwarnings('ignore')



def seed_everything(seed=0):
    """
    Function to make reproducible project
    
    Args:
        seed (int): number of seed project
    Returns:
        None
    
    """
    random.seed(seed)
    np.random.seed(seed)
    
def columns_to_str(df,columns):
    """
    Function to make columns as string
    
    Args:
        df (dataframe): dataset to make changes
        columns (list): list of names to make changes
    
    Returns:
        df (dataframe): dataset changed
        
    """
    for name in columns:
        df[name] = df[name].astype(str)
    return df

def columns_to_int(df,columns):
    """
    Function to make columns as integer
    
    Args:
        df (dataframe): dataset to make changes
        columns (list): list of names to make changes
    
    Returns:
        df (dataframe): dataset changed
        
    """
    for name in columns:
        df[name] = df[name].astype(int)
    return df






def fe_dates(name,df):
    """
    Function to make feature engineering of date
    
    Args:
        name     (str): Name of date column
        df (dataframe): dataset with date column
    Returns:
        df (dataframe): dataset with new features
    
    """
    df[name] = pd.to_datetime(df[name], errors='coerce')
    
    df['year']  = df[name].dt.year
    df['month'] = df[name].dt.month
    df['day']   = df[name].dt.day
    
    
    df['day_of_week'] = df[name].dt.day_of_week
    df['day_of_year'] = df[name].dt.day_of_year
    
    df['is_year_start']    = df[name].dt.is_year_start
    df['is_quarter_start'] = df[name].dt.is_quarter_start
    df['is_month_start']   = df[name].dt.is_month_start
    df['is_month_end']    = df[name].dt.is_month_end
    
    return df




def reduce_mem_usage(df):
    """ 
    Function to iterate through all the columns of a dataframe and modify the data type to reduce memory usage.     
    
    Args:
        df (dataframe): dataset to reduce memory
        
    Returns:
       df  (dataframe): dataset changed
       
    """
    start_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    
    for col in df.columns:
        col_type = df[col].dtype
        
        if col_type != object:
            if str(col_type) == numerics:
                c_min = df[col].min()
                c_max = df[col].max()

                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            if str(col_type)[:5] == 'float':
                c_min = df[col].min()
                c_max = df[col].max()

                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            df[col] = df[col].astype('category')

    end_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
    
    print('*'*20)
    print('*'*20)

    return df

        

## Merging by concat to not lose dtypes
def merge_by_concat(df1, df2, merge_on):
    """
    Function to merge 2 different dataset base on columns
    
    Args:
        df1  (dataframe): dataset base 
        df2  (Dataframe): dataset of new features
        merge_on  (list): list of names to make the merge
    
    Returns:
        df1  (dataframe): dataset with new features
    
    """
    merged_gf = df1[merge_on]
    merged_gf = merged_gf.merge(df2, on=merge_on, how='left')
    new_columns = [col for col in list(merged_gf) if col not in merge_on]
    merged_gf.reset_index(inplace=True,drop=True)
    df1.reset_index(inplace=True,drop=True)
    df1 = pd.concat([df1, merged_gf[new_columns]], axis=1)
    return df1


def join_columns_string(data,columns):
    """
    Function to join columns string in single column
    
    Args:
        data  (dataframe): dataset base
        columns    (list): list of names to join
        
    Results:
        auxiliat (dataframe): dataset of new column 
        
    """
    auxiliar = None
    for column in columns:
        if auxiliar is None:
            auxiliar = data[column].astype(str)
        else:
            auxiliar += '_' +data[column].astype(str)        
    return auxiliar

def get_store_item(df,modelo,punto_venta,gama):
    '''
    Function to Filter data by modelo, punto de venta and gamma
    
    Args:
        df       (dataframe): dataset base
        modelo         (str): name of modelo
        punto_venta    (str): name of punto de venta 
        gama           (str): name of gama 
    
    Returns:
        
        sub_data (Dataframe): sub group of data by modelo, punto de venta and gama
    
    '''
    
    return df[(df['Z_MODELO']==modelo) & (df['Z_PUNTO_VENTA']==punto_venta)& (df['Z_GAMA']==gama)]




def my_df_describe(df,name = 'dataset',show = True,path='',save=False):
    '''
    Function to describe categorical and numeric values
    
    Args:
        df  (dataframe): dataset base
        name      (str): name of the dataset
        show     (bool): boolean to show describe or no
        path      (Str): path to save describe of data
        save     (bool): boolean to save describe in csv
    Returns:
        numeric_desc     (dataframe): describe of numeric columns
        categorical_desc (Dataframe): describe of categorical columns
    '''
    
    path_analisis    = ''
    path_description = ''
    if save:
        path_analisis        = os.path.join(path,'analysis')
        path_description = os.path.join(path_analisis,'description') 
        for folder in [path,path_analisis,path_description]:
            if not os.path.isdir(folder):
                os.mkdir(folder)

    print(20*'*'+name+20*'*')
    objects = []
    numerics = []
    for c in df:
        if (str(df[c].dtype) in ['str','object','category','datetime64[ns]','bool']):
            objects.append(c)
        else:
            numerics.append(c)
    
    df = df.replace(to_replace=['',' ','None','NaN'], value=np.nan)
    numeric_desc, categorical_desc = None,None
    
    if len(numerics)>0:
        print('shape numerics : ',len(numerics))
        numeric_desc = df[numerics].describe().transpose()
        counting = df.nunique().transpose()
        numeric_desc['unique'] = counting[numerics]
        print('shape numeric_desc : ',numeric_desc.shape)
        numeric_desc['nulls'] = df[numerics].isna().sum().values
        numeric_desc['nulls_perc'] = df[numerics].isna().sum().values/df.shape[0]
        if save:
            filename = os.path.join(path_description,name+'_numeric_variables_description.csv')
            print('saving ',filename)
            numeric_desc.to_csv(filename)#,index=None

    if len(objects)>0:
        categorical_desc = df[objects].describe().transpose()
        #print(df[objects].isna().sum())
        #print(categorical_desc.shape)
        categorical_desc['nulls'] = df[objects].isna().sum().values
        categorical_desc['nulls_perc'] = df[objects].isna().sum().values/df.shape[0]
        if save:
            filename = os.path.join(path_description,name+'_categorical_variables_description.csv')
            print('saving',filename)
            categorical_desc.to_csv(filename)
    
    print('shape ',df.shape)
    if show:
        if len(numerics)>0:
            print(10*'*'+'numerics'+10*'*')
            display(numeric_desc)
        if len(objects)>0:
            print(10*'*'+'categorical'+10*'*')
            display(categorical_desc)
    
    return numeric_desc, categorical_desc

def get_perceptiles(values,mini=25,maxi=75):
    '''
    Function to get percentiles of a vector
    
    Args:
        values   (np.array): vector of values
        mini          (int): minimum value of percentile
        maxi          (int): maximum value of percentile
    
    Returns:
        Q1      (float): percentile 1
        Q3      (float): percentile 2
    '''
    Q1,Q3=np.percentile(values,[mini,maxi])
    return Q1,Q3
