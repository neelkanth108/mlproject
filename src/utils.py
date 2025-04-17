import os
import sys
import pandas as pd
import numpy as np
import dill
from src.exception import CustomeException
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error
def save_object(file_path,obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path,exist_ok=True)
            ## dill is used to create pickle file
        with open(file_path,"wb") as file_obj:
            dill.dump(obj,file_obj)

    except Exception as e:
        raise CustomeException(e,sys)
    
def evaluate_model(X_train,Y_train,X_test,Y_test,models):
    try:
        report={}
        for i in range(len(list(models))):
            model = list(models.values())[i]
            model.fit(X_train,Y_train)
            y_pred_train = model.predict(X_train)
            y_pred_test = model.predict(X_test)
            train_score = r2_score(Y_train,y_pred_train)
            test_score = r2_score(Y_test,y_pred_test)

            report[list(models.keys())[i]] = test_score

        return report

        
    except Exception as e:
        raise CustomeException(e,sys)
