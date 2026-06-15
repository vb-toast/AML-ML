import os
import sys
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PowerTransformer, FunctionTransformer, OneHotEncoder, TargetEncoder
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', 'preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
    
    def get_data_transformer_object(self):
        '''
        This function is responsible for data transformation and pipeline intiation
        '''

        try:
            numerical_columns = ["Amount_Received", "Amount_Paid", "From_Bank", "To_Bank"]
            low_card_categorical_columns = ["Receiving_Currency", "Payment_Currency", "Payment_Format"]
            high_card_categorical_columns = ["Send_Account", "Receive_Account"]
            
            num_pipeline = Pipeline(
                steps = [
                    ("imputer", SimpleImputer(strategy = "median")), #fills in missing values with median of each column
                    ("clipper", FunctionTransformer(
                        lambda X: np.clip(X, np.percentile(X, 1, axis = 0), np.percentile(X, 99, axis = 0)) # brings outliers to the 1st and 99th percentils of values
                    )),
                    ("power_transform", PowerTransformer(method = "yeo-johnson")), #reshapes toward a guassian distribution 
                    ("variance_filter", VarianceThreshold(threshold = 0.01)) #drops columns whose variance is below 0.01
                ]
            )
            logging.info(f"Numerical columns encoding completed, numerical columns:{numerical_columns}")

            low_card_pipeline = Pipeline(
                steps = [
                    ("imputer", SimpleImputer(strategy = "most_frequent")), #handles missing values
                    ("encoder", OneHotEncoder(
                        handle_unknown = "ignore", # ignores unseen categories on inference
                        sparse_output = False # returns dense array
                    ))
                ]
            )

            logging.info(f"Low cardinality categorical columns encoding completed, categorical columns:{low_card_categorical_columns}")

            high_card_pipeline = Pipeline(
                steps = [
                    ("imputer", SimpleImputer(strategy = "most_frequent")),
                    ("encoder", TargetEncoder(smooth = "auto"))
                ]
            )

            logging.info(f"High cardinality categorical columns encoding completed, categorical columns:{high_card_categorical_columns}")

            preprocessor = ColumnTransformer( 
                [
                    ("num_pipeline", num_pipeline, numerical_columns),
                    ("low_card_pipeline", low_card_pipeline, low_card_categorical_columns),
                    ("high_card_pipeline", high_card_pipeline, high_card_categorical_columns)
                ]
            )
            
            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)
        
    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read, train, and test data completed")
            logging.info("Obtaining preprocessing object")

            preprocessing_obj = self.get_data_transformer_object()

            target_column_name = "Is_Laundering"

            input_feature_train_df = train_df.drop(columns = [target_column_name])

            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns = [target_column_name])

            target_feature_test_df = test_df[target_column_name]

            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df, target_feature_train_df)

            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]

            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            #save the preprocessor
            save_object(
                file_path = self.data_transformation_config.preprocessor_obj_file_path,
                obj = preprocessing_obj
            )

            logging.info("Saved preproecssing object")

            return(train_arr, test_arr, self.data_transformation_config.preprocessor_obj_file_path)

        except Exception as e:
            raise CustomException(e, sys)