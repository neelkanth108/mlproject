import os , sys
from sklearn.ensemble import AdaBoostRegressor,RandomForestRegressor,GradientBoostingRegressor
from sklearn.linear_model import LinearRegression,Ridge,Lasso
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import RandomizedSearchCV,GridSearchCV
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

from dataclasses import dataclass
from src.exception import CustomeException
from src.logger import logging

from src.utils import save_object,evaluate_model

@dataclass
class ModelTrainerConfig:
    trained_model_file_path:str=os.path.join('artifacts','model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
    def initiate_model_trainer(self,train_arr,test_arr):
        try:
            logging.info("Splitting training and test data")
            x_train,y_train,x_test,y_test = (
                train_arr[:,:-1],
                train_arr[:,-1],
                test_arr[:,:-1],
                test_arr[:,-1]
            )
            models={
            'linearReg':LinearRegression(),
            'ridge':Ridge(),
            "Lasso": Lasso(),
            "K-Neighbors Regressor": KNeighborsRegressor(),
            "Decision Tree": DecisionTreeRegressor(),
            "Random Forest Regressor": RandomForestRegressor(),
            "XGBRegressor": XGBRegressor(),
            "CatBoosting Regressor": CatBoostRegressor(verbose=False),
            "AdaBoost Regressor": AdaBoostRegressor()
            }
            model_report:dict=evaluate_model(X_train=x_train,Y_train=y_train,X_test = x_test,Y_test = y_test,models=models)

            ### to get best model score
            best_model_score = max(sorted(model_report.values()))
            ## to get best model name on best score
            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]

            best_model = models[best_model_name]
            if best_model_score<0.6:
                raise CustomeException("no best model found")
            logging.info("Best found model on both training and testing dataset")
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj = best_model
            )
            predicted  = best_model.predict(x_test)
            r2_s = r2_score(y_test,predicted)

            return r2_s

        except Exception as e:
            raise CustomeException(e,sys)
        

