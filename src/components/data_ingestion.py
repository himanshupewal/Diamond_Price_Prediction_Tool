## reading data and train-test split and returning it

import os
import sys 
from src.logger import logging
from src.exception import CustomException
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass


## intialize the data ingestion config  ----  config all parameters ,files
@dataclass
class DataIngestionconfig:
    train_data_path = os.path.join('artifacts','train.csv')
    test_data_path = os.path.join('artifacts','test.csv')

    raw_data_path = os.path.join('artifacts','raw.csv')


## create a data ingestion class --- every steps to do ingestion

class DataIngestion:

    def __init__(self):
        self.ingestion_config = DataIngestionconfig()


    def initiate_ingestion(self):

        logging.info('Data ingestion has started')

        try:
            df = pd.read_csv(os.path.join('Notebook/Data','gemstone.csv'))
            logging.info('Data read as pandas Frame')


            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path),exist_ok=True)

            df.to_csv(self.ingestion_config.raw_data_path,index=False)

            logging.info('raw file created')
            logging.info('Train slpit has started')

            train_set,test_set = train_test_split(df,test_size=0.3,random_state=42)
            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header =True)
            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)

            logging.info('Ingestion has completed')


            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )


        except Exception as e:
            logging.info("Error in initiating data ingestion config")