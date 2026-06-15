import os
import sys
from dataclasses import dataclass
from catboost import CatBoostRegressor
from sklearn.ensemble import(
    GradientBoostingClassifier,
    RandomForestClassifier
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBRegressor
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
    
    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Splitting, training, and testing input data")
            x_train, y_train, x_test, y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            #use regressor models to predict Is_Laundering, which is binary, either 0 - False, 1 - True
            models = { 
                "Random Forest": RandomForestClassifier(),
                "Decision Tree": DecisionTreeClassifier(),
                "Gradient Boosting": GradientBoostingClassifier(),
                "XGBRegressor": XGBRegressor(),
                "CatBoosting Regressor": CatBoostRegressor(verbose=False),
            } 
            
            params = {
                "Decision Tree": {
                    'criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    # 'splitter':['best','random'],
                    # 'max_feature':['sqrt','log2'],
                },
                "Random Forest":{
                    #'criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    # 'max_feature':['sqrt','log2'],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Gradient Boosting": {
                    # 'loss':p'squared_error','huber','absolute_error','quantile'],
                    'learning_rate': [0.1,0.01,0.05,0.001],
                    'subsample': [0.6,0.7,0.75,0.8,0.85,0.9],
                    #'criterion':['squared_error','friedman_mse'],
                    #'max_features':['auto','sqrt','log2'],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Linear Regression":{},
                "K-Neighbors Regressor":{
                    'n_neighbors':[5,7,9,11],
                    #'weights':['uniform','distance'],
                    #'algorithm':['ball_tree','kd_rree','brute']
                },
                "XGBRegressor":{
                    'learning_rate': [0.1,0.01,0.05,0.001],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "CatBoosting Regressor":{
                    'depth': [6,7,10],
                    'learning_rate': [0.01,0.05,0.1],
                    'iterations': [30,50,100]
                },
                "AdaBoost Regressor": {
                    'learning_rate': [0.1,0.01,0.05,0.001],
                    #'loss':['linear','square','exponential'],
                    'n_estimators': [8,16,32,64,128,256]
                }
            }

            model_report:dict = evaluate_models(
                x_train = x_train, 
                y_train = y_train, 
                x_test = x_test, 
                y_test = y_test, 
                models = models,
                param = params
            )

            best_model_score = max(sorted(model_report.values()))

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            best_model = models[best_model_name]

            if best_model_score < 0.6:
                raise CustomException("No model fits (Based on 60% score)")
            
            logging.info("Best found model on both training and testing dataset")

            logging.info(f"Best model name: {best_model}, best model score: {best_model_score}")

            save_object(
                file_path = self.model_trainer_config.trained_model_file_path,
                obj = best_model
            ) 

            predicted = best_model.predict(x_test)

            r2_square = r2_score(y_test, predicted)
            return r2_square

        except Exception as e:
            raise CustomException(e, sys)

