import os,sys
import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression,Lasso,Ridge,ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from src.exception import CustomException
from sklearn.ensemble import RandomForestRegressor
from src.logger import logging
from src.utils import save_obj
from src.utils import eval_model
from dataclasses import dataclass


@dataclass

class ModelTrainerConfig:

    train_model_file_path = os.path.join('artifacts','model.pkl')


class ModelTrainer:

    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()


    def initate_model_training(self,train_array,test_array):

        try:

            logging.info("splititng train - test datasets")


            X_train,y_train,X_test,y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )


            models = {
                'LinearRegression':LinearRegression(),
                'Lasso':Lasso(),
                'Ridge':Ridge(),
                'ElasticNet':ElasticNet(),
                'DecisionTree':DecisionTreeRegressor(),
                'SVR':SVR(),
                "RandomForest":RandomForestRegressor(),
                'XGboost':XGBRegressor()
        }
            

            model_report :dict=eval_model(X_train,y_train,X_test,y_test,models)
            print(model_report)

            print("\n================================================================================\n")

            logging.info(f'Model Report : {model_report}')


            best_model_score  = max(sorted(model_report.values()))

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)

            ]

            best_model = models[best_model_name]

            print(f"Best Model is {best_model_name} with R2_score {best_model_score}")
            print('\n==================================================================================\n')
            logging.info(f"Best Model is: {best_model_name} with R2_score :{best_model_score}")

            save_obj(
                file_path=self.model_trainer_config.train_model_file_path,
                obj=best_model
            )




        except Exception as e:
            logging.info("Error in model training")
            raise CustomException(e,sys)

