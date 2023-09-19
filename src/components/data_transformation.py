import os 
import sys 
import numpy as np
import pandas as pd
from dataclasses import dataclass
from src.exception import CustomException
from src.logger import logging

## transformation 
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OrdinalEncoder

## Pipelines 
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from src.utils import save_obj
#from src.pipeline.training_pipeline import test_data_path,train_data_path



## Data Transform Config

@dataclass

class DataTransformtionconfig:
    preprocessor_obj_file_path = os.path.join('artifacts','preprocessor.pkl')




## Data Trans class

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformtionconfig()



    def get_data_transformation(self):

        try:
            logging.info('preprocessing has Initiated')

            # Define which columns should be ordinal-encoded and which should be scaled
            categorical_cols = ['cut', 'color','clarity']
            numerical_cols = ['carat', 'depth','table', 'x', 'y', 'z']


            # Define the custom ranking for each ordinal variable
            cut_cat = ['Fair', 'Good', 'Very Good','Premium','Ideal']
            color_cat = ['D', 'E', 'F', 'G', 'H', 'I', 'J']
            clarity_cat = ['I1','SI2','SI1','VS2','VS1','VVS2','VVS1','IF']


            logging.info(' Initiated with pipeline')


            numarical_pipeline = Pipeline(
                steps=[
                    #         ('imputer', SimpleImputer()), 
                    ('imputer',SimpleImputer(strategy='median')),
                    ('scalar',StandardScaler())

                ]
            )


            categorical_pipepline = Pipeline(
                steps=[
                    ('imputer',SimpleImputer(strategy='most_frequent')),
                    ('ordinal_encoding',OrdinalEncoder(categories=[cut_cat,color_cat,clarity_cat])),
                    ('scalar',StandardScaler()),

                ]
            )



            preprocessor = ColumnTransformer(
                [
                    ('numarical_pipepline',numarical_pipeline,numerical_cols),
                    ('categorical_pipeline',categorical_pipepline,categorical_cols)
                ]
            )

            return preprocessor

        except Exception as e:
            logging.info('Error in data Transformation')
            raise CustomException(e,sys)


    def initiate_data_transformation(self,train_path,test_path):

        try:

            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info('reading completed')
            logging.info(f'Train Dataframe Head : \n{train_df.head().to_string()}')
            logging.info(f'Test Dataframe Head : \n{test_df.head().to_string()}')


            logging.info("Obtainig preprocessing object")

            preprocession_obj = self.get_data_transformation()


            target_col_name = 'price'

            drop_cols = [target_col_name,'id']

            input_features_train_df = train_df.drop(columns=drop_cols,axis=1)
            target_feature_train_df  = train_df[target_col_name]

            input_features_test_df = test_df.drop(columns=drop_cols,axis=1)
            target_feature_test_df  = test_df[target_col_name]


            #### applying transformation


            input_features_train_arr = preprocession_obj.fit_transform(input_features_train_df)

            
            input_features_test_arr = preprocession_obj.transform(input_features_test_df)

            logging.info('Applying preprocessing obj on training and testing dataset')


            train_arr = np.c_[input_features_train_arr,np.array(target_feature_train_df)]
            test_arr = np.c_[input_features_test_arr,np.array(target_feature_test_df)]

            save_obj(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocession_obj
            )


            logging.info("Processor pickle created ")


            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )
        except Exception as e:
            logging.info('Error in Transformation Ingestation')
            raise CustomException(e,sys)





