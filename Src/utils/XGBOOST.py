import xgboost as xgb
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

class my_xgboost:
    
    def __init__(self,model_name):
        self.model_name = model_name
        
    def train(self,X_train,y_train,X_validation,y_validation):
        objects = []
        numerics = []
        for c in X_train:
            if (str(X_train[c].dtype) in ['str','object','category','datetime64[ns]','bool']):
                objects.append(c)
            else:
                numerics.append(c)
        
        self.objects  = objects
        self.numerics = numerics
        self.columns = X_train.columns
        
        self.enconders = {}
        
        self.X_train = X_train
        self.X_validation = X_validation
        for column in self.objects:

            le = LabelEncoder()

            le.fit(np.append(pd.concat([X_train[column],X_validation[column]]),np.nan))
            
            self.enconders[column] = le
            X_train[column]      = le.transform(X_train[column])
            X_validation[column] = le.transform(X_validation[column])



        self.model = xgb.XGBRegressor(base_score=0.25, booster='gbtree', colsample_bylevel=1,#num_boost_round=100,
       colsample_bytree=1, gamma=0, learning_rate=0.015, max_delta_step=0,
       max_depth=7, min_child_weight=1, missing=1, n_estimators=2000,verbose_eval=20,
       n_jobs=8, nthread=8, objective='reg:tweedie', random_state=0,tweedie_variance_power=1.07,
       reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=42,tree_method='gpu_hist',
       silent=False, subsample=1)
        
        
        self.model_result = self.model.fit(X_train._get_numeric_data().values,y_train,
                verbose=20,
                early_stopping_rounds=40,
                eval_metric='rmse',
                eval_set=[(X_train._get_numeric_data().values,y_train),(X_validation._get_numeric_data().values,y_validation)])

        self.best_iteration = self.model.get_booster().best_ntree_limit

        return self.model
    
    def predict(self,X_data):
        
        for column in self.objects:
            le = LabelEncoder()

            le.fit(np.append(pd.concat([self.X_train[column],self.X_validation[column]]),np.nan))
            new = list(set(X_data[column])-set(le.classes_))

            for element in new:
                le.classes_ = np.append(le.classes_, element)
            X_data[column]      = le.transform(X_data[column])
            
        return np.maximum(self.model.predict(X_data.values, ntree_limit=self.best_iteration) ,0)
    
    def model_save(self,path):
        self.model.save_model(path)
    
    def get_feature_importance(self):
        df_feature_importance = pd.DataFrame()
        df_feature_importance["feature"] = self.columns
        df_feature_importance["importance"] = self.model.feature_importances_
        df_feature_importance = df_feature_importance.sort_values(['importance'],ascending=False)
        return df_feature_importance
