# import custom  libraries
from utils import submission_analysis
from utils import XGBOOST

# import python libraries
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import mean_absolute_error

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os



PATH_RESULTS = None
SHIFT = None
TARGET = None

def define_parameters(PATH_RESULTS,SHIFT,TARGET="Demanda"):
    """
    Define parameters in training proccess to save models and predictions in folder
    
    Args:
    
        PATH_RESULTS (str): path to save results as models and predictions
        SHIFT        (int): the number of week to analysis ( SHIFT =1 to analyse WEEK 1)
        TARGET       (str): the name of the target variable (TARGET = Demanda)
        
    Returns:
        None
        
    """
    PATH_RESULTS = PATH_RESULTS
    SHIFT = SHIFT
    TARGET = TARGET
    submission_analysis.TARGET = TARGET
    
def create_path(path):
    """
    Function to create a root of path recursively
    
    Args:
        path (str): path to create folder
        
    Returns:
        None
    
    """
    print('create folder',path)
    directory = ''
    for sub_path in path.split('/'):
        directory = os.path.join(directory,sub_path)
        if not os.path.exists(directory):
            os.makedirs(directory)
            


def analysis_model(X_train,y_train,X_validation,y_validation,X_test,y_test,model):
    """
    Function to get metrics and evaluate fitting with validation and test RMSE
    
    Args:
        X_train      (dataframe): dataset to training
        X_validation (dataframe): dataset to validation
        X_test       (dataframe): dataset to testing
        y_train      (dataframe): target of training
        y_validation (dataframe): target of validation
        y_test       (dataframe): target of testing
        model     (model object): model to make inference (xgboost model)
        
    Returns:
        X_validation_pred (dataframe): validation prediction resullt 
        X_test_pred       (dataframe): testing prediction result
        df_val_rmse       (float)    : RMSE of validation
        df_test_rmse      (float)    : RMSE of testing
        
    """
    print('Doing predictions...')
    #X_train_pred      = model.predict(X_train) # uncomment if the speed is no neccesary
    X_validation_pred = model.predict(X_validation)
    X_test_pred       = model.predict(X_test)
    print('Doing metrics ... ')
    #df_train_rmse = mean_squared_error(y_train,X_train_pred, squared=False)
    df_val_rmse   = mean_squared_error(y_validation,X_validation_pred, squared=False)
    df_test_rmse  = mean_squared_error(y_test,X_test_pred, squared=False)
    
    df_val_mape   = mean_absolute_percentage_error(y_validation[y_validation>0],X_validation_pred[y_validation>0])
    df_test_mape  = mean_absolute_percentage_error(y_test[y_test>0],X_test_pred[y_test>0])
    
    df_val_mase   = mean_absolute_error(y_validation,X_validation_pred)
    df_test_mase  = mean_absolute_error(y_test,X_test_pred)
    
    print('Final score mean_squared_error')
    #print('Score train:',df_train_rmse)
    print('Score val  :',df_val_rmse)
    print('Score test :',df_test_rmse)
    #mean_absolute_percentage_error
    
    print('Final score mean_absolute_percentage_error')
    #print('Score train:',df_train_rmse)
    print('Score val  :',df_val_mape)
    print('Score test :',df_test_mape)
    
    print('Final score mean_absolute_error')
    #print('Score train:',df_train_rmse)
    print('Score val  :',df_val_mase)
    print('Score test :',df_test_mase)
    print('end !')
    #return X_train_pred,X_validation_pred,X_test_pred,df_train_rmse,df_val_rmse,df_test_rmse
    return X_validation_pred,X_test_pred,df_val_rmse,df_test_rmse

def save_feature_importance(path,df_feature_importance):
    """
    Function to plot and save feature importance
    
    Args:
        path                        (str): path to save image of feature importance
        df_feature_importance (dataframe): dataframe of importance for each variable in the model 
    
    Returns:
        None
        
    """
    print('saving feature importance ..',path)
    plt.figure(figsize=(14,40))
    sns.barplot(x="importance",y="feature",data=df_feature_importance)
    plt.tight_layout()
    plt.title(' feature importance')
    plt.savefig(path)
    plt.close()

def save_test_prediction(path,X_test_pred,y_test):
    """
    Function to save plot of prediction of testing prediction and real values to evaluation overfitting in training
    
    Args:
        path              (str): path to save testing predictions an real values 
        X_test_pred (dataframe): dataframe of testing predictions
        y_test      (dataframe): dataframe of testing real values
    
    Returns:
        None
        
    """
    print('saving test prediction ..',path)
    fig = plt.figure(figsize=(20,4))
    plt.plot(X_test_pred,'b', alpha=0.7)
    plt.plot(y_test.values,'r', alpha=0.4)
    plt.legend(["test prediction", "test real"], loc ="upper right")
    plt.title(' test prediction')
    plt.savefig(path)
    plt.close()
    


def training_model(model_type,model_version,X_train,y_train,X_validation,y_validation,X_test,y_test,X_submission,df_test,result):
    """
    Function to train model and evaluate fitting
    
    
    Args:
        model_type         (str): model name to make training (xgbost)
        model_version      (str): model version name to save training and make different experiments
        X_train      (dataframe): dataset to training
        X_validation (dataframe): dataset to validation
        X_test       (dataframe): dataset to testing
        y_train      (dataframe): target of training
        y_validation (dataframe): target of validation
        y_test       (dataframe): target of testing
        df_test      (dataframe): dataset to submission (unseen data)
        result       (dataframe): dataset of last inference in submission  (predicted data)
        
    Returns:
    
        X_validation_pred (dataframe): validation prediction resullt 
        X_test_pred       (dataframe): testing prediction result
        df_submission     (dataframe): submission prediction result
        feature_importance(dataframe): feature importance of each value in the training 
    
    """

    directory_model = os.path.join(PATH_RESULTS,model_type,model_version,'shift_'+str(SHIFT))
    
    print('directory_model = ',directory_model)
    create_path(directory_model)
    
    ###### training ############

    model_name = model_version+'_'+model_type     
    if model_type == 'xgboost':
        model = XGBOOST.my_xgboost(model_name)

    _ = model.train(X_train,y_train,X_validation,y_validation)

    ###### analisis training #############


    X_validation_pred,X_test_pred,df_val_rmse,df_test_rmse = analysis_model(X_train,y_train,X_validation,y_validation,X_test,y_test,model)

    ###### save analisis training  ############
    metrics_name = 'rmse_val_'+str(np.round(df_val_rmse,4))+'_test_'+str(np.round(df_test_rmse,4))

    model_name_to_save         = metrics_name+'_model.pkl'
    results_name_to_save       = metrics_name+'_prediction_test.png'
    feature_importance_to_save = metrics_name+'_feature_importance.png'
    feature_importance_csv_to_save = metrics_name+'_feature_importance.csv'

    path_model   = os.path.join(directory_model,model_name_to_save)
    path_results_test = os.path.join(directory_model,results_name_to_save)
    path_feature_importance = os.path.join(directory_model,feature_importance_to_save)
    path_feature_importance_csv = os.path.join(directory_model,feature_importance_csv_to_save)

    save_feature_importance(path_feature_importance,model.get_feature_importance())
    save_test_prediction(path_results_test,X_test_pred,y_test)
    
    model.get_feature_importance().to_csv(path_feature_importance_csv,index=None)
    ###### save model  #############

    print('saving model..',path_model)
    model.model_save(path_model)

    ###### submission prediction #############
    X_submission_pred = model.predict(X_submission)


    #### make submission #########
    df_submission = submission_analysis.get_submission(df_test,X_submission_pred)
    df_submission[TARGET].min(),df_submission[TARGET].max()
    df_submission = df_submission.merge(result[['Z_MODELO','Z_PUNTO_VENTA','Z_GAMA','Z_WEEK',TARGET+'_real']],how='left')


    #### save submission #########
    submission_name_to_save        = metrics_name +'_submission.csv'
    path_submission      = os.path.join(directory_model,submission_name_to_save)
    rmse_target = submission_analysis.save_submission(df_submission,df_val_rmse,df_test_rmse,path_submission,save = True)

    #### save submission analysis #########
    submission_plot_name_to_save   = metrics_name +'_prediction_submission.png'
    path_submission_plot = os.path.join(directory_model,submission_plot_name_to_save)
    submission_analysis.save_plot_submission(df_submission,path_submission_plot)

    ###### save submission analysis weeks ############
    submission_half_plot_name_to_save   = metrics_name +'_prediction_submission_half.png'
    path_half_submission      = os.path.join(directory_model,submission_half_plot_name_to_save)
    submission_analysis.save_plot_submission_half(df_submission,path_half_submission)

    submission_weeks_plot_name_to_save   = metrics_name +'_prediction_submission_weeks.png'
    path_weeks_submission      = os.path.join(directory_model,submission_weeks_plot_name_to_save)
    submission_analysis.save_plot_submission_weeks(df_submission,path_weeks_submission)

    
    submission_total_weeks_plot_name_to_save   = metrics_name +'_prediction_submission_total_weeks.png'
    path_total_weeks_submission      = os.path.join(directory_model,submission_total_weeks_plot_name_to_save)
    submission_analysis.save_plot_submission_total_weeks(df_submission,path_total_weeks_submission)
    
    return X_validation_pred,X_test_pred,df_submission,model.get_feature_importance()





def training_model_cv(model_type,model_version,X_train,y_train,X_test,y_test,X_submission,df_test,result,N_FOLDS,X_train_Z_WEEK):
    """
    Function to make Cross Validation of different models base of N_FOLDS  and evaluate fitting
    
    
    Args:
        model_type         (str): model name to make training (xgbost)
        model_version      (str): model version name to save training and make different experiments
        X_train      (dataframe): dataset to training
        X_test       (dataframe): dataset to testing
        y_train      (dataframe): target of training
        y_test       (dataframe): target of testing
        df_test      (dataframe): dataset to submission (unseen data)
        result       (dataframe): dataset of last inference in submission  (predicted data)
        N_FOLDS            (int): number of folds in cross validation
        
    Returns:
 
        df_submission      (dataframe): submission prediction result (mean of each fold)
        feature_importance (dataframe): feature importance of cross validation (mean of each fold)
    
    """
    objects = []
    numerics = []
    for c in X_train:
        if (str(X_train[c].dtype) in ['str','object','category','datetime64[ns]','bool']):
            objects.append(c)
        else:
            numerics.append(c)

    for column in objects:
        X_train[column] = X_train[column].astype('category')
        X_test[column]  = X_test[column].astype('category')
        X_submission[column]  = X_submission[column].astype('category')

    
    #y_stratified = pd.cut(y_train, bins=10, labels=False)
    #folds = StratifiedKFold(n_splits=5,shuffle=True,random_state=0)
        
    folds = KFold(n_splits = N_FOLDS, shuffle = True, random_state = 100)
    print(folds)


    fold_pred = np.zeros(len(X_train))
    feature_importance_df = pd.DataFrame()
    predictions_submission = pd.DataFrame()
    predictions_test       = pd.DataFrame()
    lgb_preds = np.zeros(len(X_submission))

    for fold_, (trn_idx, val_idx) in enumerate(folds.split(y_train,y_train)):

        print(20*'*'," fold n°{}".format(fold_))

        X_train_cv,y_train_cv           = X_train.iloc[trn_idx],y_train.iloc[trn_idx]
        X_validation_cv,y_validation_cv = X_train.iloc[val_idx],y_train.iloc[val_idx]

        print('Training ...')

        model_version_fold = model_version+'/fold_'+str(fold_+1)

        X_validation_pred,X_test_pred,df_submission,df_feature_importance = training_model(model_type,model_version_fold,X_train_cv,y_train_cv,
                                                                      X_validation_cv,y_validation_cv,X_test,y_test,X_submission,df_test,result)

        fold_pred[val_idx] = X_validation_pred
        df_feature_importance["fold"] = fold_ + 1
        feature_importance_df = pd.concat([feature_importance_df, df_feature_importance], axis=0)


        print('Predict submission ')
        X_submission_pred = df_submission[TARGET]
        lgb_preds += X_submission_pred / (N_FOLDS)

        predictions_submission[str(fold_ + 1)] = X_submission_pred
        predictions_test[str(fold_ + 1)] = X_test_pred

    print("CV score: {:<8.5f}".format(np.sqrt(mean_squared_error(fold_pred, y_train))))
    
    fold_prediction_test = predictions_test.mean(axis=1)
    
    cv_train_rmse = np.sqrt(mean_squared_error(fold_pred, y_train))
    cv_test_rmse  = np.sqrt(mean_squared_error(fold_prediction_test, y_test))
    
    #mean_absolute_percentage_error
    
    cv_train_mape = mean_absolute_percentage_error(y_train[y_train>0], fold_pred[y_train>0])
    cv_test_mape  = mean_absolute_percentage_error(y_test[y_test>0],fold_prediction_test.values[y_test>0])
    cv_train_mase = mean_absolute_error(y_train, fold_pred)
    cv_test_mase  = mean_absolute_error(y_test,fold_prediction_test)
    
    df_cv_results = pd.DataFrame(y_train.values,columns=['y_train_real'])
    df_cv_results['y_train_pred'] = fold_pred
    #df_cv_results['y_test'] = y_test
    #df_cv_results['y_test_pred'] = fold_prediction_test.values
    df_cv_results['Z_MARCA'] = X_train['Z_MARCA'].values
    df_cv_results['Z_GAMA'] = X_train['Z_GAMA'].values
    df_cv_results['Z_MODELO'] = X_train['Z_MODELO'].values
    df_cv_results['Z_DEPARTAMENTO'] = X_train['Z_DEPARTAMENTO'].values
    df_cv_results['Z_PUNTO_VENTA'] = X_train['Z_PUNTO_VENTA'].values
    df_cv_results['Z_WEEK'] = X_train_Z_WEEK.values
    
    print('cv_train_rmse : ',cv_train_rmse)
    print('cv_test_rmse  : ',cv_test_rmse)
    
    print('cv_train_mape : ',cv_train_mape)
    print('cv_test_mape  : ',cv_test_mape)
    
    print('cv_train_mase : ',cv_train_mase)
    print('cv_test_mase  : ',cv_test_mase)
    
    metrics_name = 'rmse_cv_train_'+str(np.round(cv_train_rmse,4))+'_rmse_cv_test_'+str(np.round(cv_test_rmse,4))

    
    # ranking all feature by avg importance score from Kfold, select top100
    all_features = feature_importance_df[["feature", "importance"]].groupby("feature").mean().sort_values(by="importance", ascending=False)
    all_features.reset_index(inplace=True)
    important_features = list(all_features[0:30]['feature'])
    
    
    directory_model = os.path.join(PATH_RESULTS,model_type,model_version,'shift_'+str(SHIFT))
    create_path(directory_model)
    

    feature_importance_to_save = metrics_name+'_feature_importance.png'
    path_feature_importance = os.path.join(directory_model,feature_importance_to_save)

    plt.figure(figsize=(14,25))
    sns.barplot(x="importance",y="feature",data=all_features)
    plt.tight_layout()
    plt.title(' feature importance')
    plt.savefig(path_feature_importance)
    plt.close()
    
    ########## make predictions CV ################ 
    X_submission_pred = predictions_submission.mean(axis=1)
    df_submission = submission_analysis.get_submission(df_test,X_submission_pred)
    
    ########## make analysis CV ################ 

    results_name_to_save       = metrics_name+'_prediction_test.png'
    feature_importance_to_save = metrics_name+'_feature_importance.png'
    feature_importance_csv_to_save = metrics_name+'_feature_importance.csv'

    path_results_test = os.path.join(directory_model,results_name_to_save)
    path_feature_importance = os.path.join(directory_model,feature_importance_to_save)
    path_feature_importance_csv = os.path.join(directory_model,feature_importance_csv_to_save)

    save_feature_importance(path_feature_importance,all_features)
    save_test_prediction(path_results_test,X_test_pred,y_test)

    all_features.to_csv(path_feature_importance_csv,index=None)

    #### make submission CV #########
    df_submission[TARGET].min(),df_submission[TARGET].max()
    df_submission = df_submission.merge(result[['Z_MODELO','Z_PUNTO_VENTA','Z_GAMA','Z_WEEK',TARGET+'_real']],how='left')

    #### save df cv results ######
    df_cv_results_name_to_save        = metrics_name +'_df_cv_results.csv'
    path_df_cv_results_name      = os.path.join(directory_model,df_cv_results_name_to_save)
    df_cv_results.to_csv(path_df_cv_results_name,index=None)
    
    #### save submission #########
    submission_name_to_save        = metrics_name +'_submission.csv'
    path_submission      = os.path.join(directory_model,submission_name_to_save)
    rmse_target = submission_analysis.save_submission(df_submission,cv_train_rmse,cv_test_rmse,path_submission,save = True)

    #### save submission analysis #########
    submission_plot_name_to_save   = metrics_name +'_prediction_submission.png'
    path_submission_plot = os.path.join(directory_model,submission_plot_name_to_save)
    submission_analysis.save_plot_submission(df_submission,path_submission_plot)

    ###### save submission analysis weeks ############
    submission_half_plot_name_to_save   = metrics_name +'_prediction_submission_half.png'
    path_half_submission      = os.path.join(directory_model,submission_half_plot_name_to_save)
    submission_analysis.save_plot_submission_half(df_submission,path_half_submission)

    submission_weeks_plot_name_to_save   = metrics_name +'_prediction_submission_weeks.png'
    path_weeks_submission      = os.path.join(directory_model,submission_weeks_plot_name_to_save)
    submission_analysis.save_plot_submission_weeks(df_submission,path_weeks_submission)
    
    submission_total_weeks_plot_name_to_save   = metrics_name +'_prediction_submission_total_weeks.png'
    path_total_weeks_submission      = os.path.join(directory_model,submission_total_weeks_plot_name_to_save)
    submission_analysis.save_plot_submission_total_weeks(df_submission,path_total_weeks_submission)
    
    metrics = [cv_train_rmse,cv_test_rmse,cv_train_mape,cv_test_mape,cv_train_mase,cv_test_mase]
    
    return df_submission,all_features,metrics