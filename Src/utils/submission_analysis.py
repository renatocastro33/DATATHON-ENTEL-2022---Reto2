import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import pandas as pd
import seaborn as sns

TARGET = None


def get_submission(df_test,prediction_value):
    df_test.reset_index(drop=True,inplace=True)
    df_test[TARGET] = prediction_value
    df_test[TARGET][df_test[TARGET]<=1e-5] = 0
    print(df_test.shape)

    base_cum = df_test[[TARGET, 'Z_WEEK_DATE']].groupby(['Z_WEEK_DATE']).sum().sort_values(TARGET , ascending = [False]).reset_index()
    base = df_test[['Z_MODELO','Z_PUNTO_VENTA','Z_GAMA','Z_WEEK','Z_WEEK_DATE','date_block_num',TARGET]].merge(base_cum[['Z_WEEK_DATE',TARGET]].rename(columns = {TARGET: TARGET+'_sum'}), on = 'Z_WEEK_DATE', how = "left")
    base[TARGET+'_scale'] = base[TARGET]*100/base[TARGET+'_sum']
    return base

def save_submission(df_submission,df_val_rmse,df_test_rmse,path,save = False):
    print('saving submission ..',path)

    rmse_target = mean_squared_error(df_submission[TARGET],df_submission[TARGET+'_real'], squared=False)

    print('Final score referencial mean_squared_error')
    print('RMSE Score referencial TARGET:',rmse_target)

    if save:
        print('saving ..')
        df_submission.to_csv(path,index=None)
        print('saved ! ')
    return rmse_target

def save_plot_submission(df_submission,path):
    fig = plt.figure(figsize=(20,8))
    plt.plot(df_submission[TARGET+'_real'],'ro', alpha=0.3, label= ' real')
    plt.plot(df_submission[TARGET],'bo', alpha=0.4,label=TARGET+' pred     ')
    plt.plot(df_submission[TARGET+'_scale'],'go', alpha=0.4, label= TARGET+' pred scale    ')
    plt.title('total target distribution')
    plt.legend( loc ="upper right")
    plt.savefig(path)
    plt.close()

    
def save_plot_submission_half(df_submission,path):
    print(('Saving Prediction first 5 and last 5 weeks ...'))
    print('*'*10)
    print("['date_block_num']<55")
    print('*'*10)

    difference = mean_squared_error(df_submission[TARGET][df_submission['date_block_num']<55],
                                    df_submission[TARGET+'_real'][df_submission['date_block_num']<55], squared=False)

    print('Final score referencial mean_squared_error')
    print('RMSE Score referencial TARGET:',difference)

    fig = plt.figure(figsize=(30,4))

    plt.subplot(1,2,1)
    plt.plot(df_submission[TARGET+'_real'][df_submission['date_block_num']<55],'ro', alpha=0.3, label= ' real')
    plt.plot(df_submission[TARGET][df_submission['date_block_num']<55],'bo', alpha=0.4,label=TARGET+' pred     ')
    plt.plot(df_submission[TARGET+'_scale'][df_submission['date_block_num']<55],'go', alpha=0.4, label= TARGET+' pred scale    ')
    plt.title("['date_block_num']<55  | RMSE = "+str(difference))

    plt.legend( loc ="upper right")

    print('*'*10)
    print("['date_block_num']>=55")
    print('*'*10)

    difference = mean_squared_error(df_submission[TARGET][df_submission['date_block_num']>=55],
                                    df_submission[TARGET+'_real'][df_submission['date_block_num']>=55], squared=False)

    print('Final score referencial mean_squared_error')
    print('RMSE Score referencial TARGET:',difference)

    plt.subplot(1,2,2)


    plt.plot(df_submission[TARGET+'_real'][df_submission['date_block_num']>=55],'ro', alpha=0.3, label= ' real')
    plt.plot(df_submission[TARGET][df_submission['date_block_num']>=55],'bo', alpha=0.4,label=TARGET+' pred     ')
    plt.plot(df_submission[TARGET+'_scale'][df_submission['date_block_num']>=55],'go', alpha=0.4, label= TARGET+' pred scale    ')
    plt.title("['date_block_num']>=55  | RMSE = "+str(difference))

    plt.legend( loc ="upper right")
    plt.suptitle('Prediction first 5 and last 5 weeks')
    plt.savefig(path)
    plt.close()


def save_plot_submission_weeks(df_submission,path):
    
    print('Saving analysis by week .. ')
    mini = df_submission['date_block_num'].min()
    maxi = df_submission['date_block_num'].max()
    fig = plt.figure(figsize=(20,40))

    for idx,i in enumerate(range(mini,maxi+1)):
        plt.subplot(10,2,idx*2+1)
        sub_real = df_submission[TARGET+'_real'][df_submission['date_block_num']==i].values
        sub_pred = df_submission[TARGET][df_submission['date_block_num']==i].values
        plt.plot(range(len(sub_real)),sub_real,'ro', alpha=0.3, label= ' real',markersize=2)
        plt.plot(range(len(sub_pred)),sub_pred,'bo', alpha=0.4,label=TARGET+' pred     ',markersize=2)
        difference = mean_squared_error(sub_real,sub_pred, squared=False)
        plt.title('week = '+str(i)+'| rmse = '+str(difference))
        plt.legend( loc ="upper right")
        plt.subplot(10,2,idx*2+2)
        my_dict = {
            'real >10 ': sub_real[sub_real>10], 
            'pred > 10':sub_pred[sub_real>10],
            'real >100 ': sub_real[sub_real>100], 
            'pred > 100':sub_pred[sub_real>100]
        }

        plt.boxplot(my_dict.values())
        plt.xticks(range(1,len(my_dict.keys())+1),my_dict.keys())
    plt.suptitle('Prediction by weeks')
    plt.savefig(path)
    plt.close()
    
def save_plot_submission_total_weeks(df_submission,path):
    total_semanal = df_submission[['Z_WEEK',TARGET+'_real',TARGET]].groupby(['Z_WEEK']).sum().reset_index()
    total_semanal_melt = pd.melt(total_semanal, id_vars =['Z_WEEK'], value_vars =[TARGET+'_real',TARGET])
    plt.figure(figsize=(15,8))
    sns.lineplot(x="Z_WEEK", y="value", data=total_semanal_melt, hue="variable",style="variable",markers=True, dashes=False)
    plt.suptitle('Prediction sum total weeks')
    plt.savefig(path)
    plt.close()