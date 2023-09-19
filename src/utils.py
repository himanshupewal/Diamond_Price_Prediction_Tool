import os,sys
import pandas as pd
import numpy as np
import pickle
from src.exception import CustomException
from src.logger import logging
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error
def save_obj(file_path,obj):

    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path,exist_ok=True)

        with open(file_path,'wb') as file_obj:
            pickle.dump(obj,file_obj)



    except Exception as e:
        raise CustomException(e,sys)
    


def eval_model(X_train,y_train,X_test,y_test,models):


    try:
        report ={}

        for i in range(len(models)):

            model = list(models.values())[i]

            model.fit(X_train,y_train)

            logging.info("Model is fit")

            y_pred = model.predict(X_test)

            logging.info("Prediction completed")

            test_model_score = r2_score(y_test,y_pred)

            logging.info(f"r2 score {test_model_score}")

            report[list(models.keys())[i]] = test_model_score

        return report
    
    except Exception as e:
        logging.info('Exception Occured in load_object function utils')
        raise CustomException(e,sys)



def load_object(file_path):

    try:

        with open(file_path,'rb') as file_object:
            return pickle.load(file_object)
        
    except Exception as e:
        logging.info("Exception occured in load_obj unitls")
        raise CustomException(e,sys)


