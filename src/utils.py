import os
import sys
import pandas as pd
import numpy as np
import dill
from src.exception import CustomException
from sklearn.metrics import average_precision_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from src.logger import logging

def save_object(file_path, obj):
    '''
    Saves fitted preprocessor to be tested on further data to be refined, scaled, and encoded
    '''
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok = True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)
        
    except Exception as e:
        raise CustomException(e, sys)
    
def evaluate_models(x_train, y_train, x_test, y_test, models, param):
    '''
    Trains, scores, and reports on multiple models against the data
    '''
    try:
        report = {}

        for i in range(len(list(models))):
            model = list(models.values())[i]

            #model.fit(x_train, y_train) # Trains the model

            para = param[list(models.keys())[i]]
            
            # use grid search on a more powerful computer
            # gs = GridSearchCV(model, para, cv = 3, scoring = "average_precision", n_jobs = -1) # scoring optimizes for AUC-PR during grid search, n_jobs parallelizes acros cpu cores
            
            gs = RandomizedSearchCV(model, para, n_iter = 20, cv = 3, scoring = "average_precision", n_jobs = -1, random_state = 42) # only tests 20 random combinations per model

            gs.fit(x_train, y_train)

            model.set_params(**gs.best_params_)

            model.fit(x_train, y_train)

            logging.info(f"{list(models.keys())[i]} has finished training with AUC-PR: {test_model_score:.4f}")

            y_train_pred = model.predict_proba(x_train)

            y_test_pred = model.predict_proba(x_test)

            train_model_score = average_precision_score(y_train, y_train_pred)

            test_model_score = average_precision_score(y_test, y_test_pred)

            report[list(models.keys())[i]] = test_model_score
        
        return report

    except Exception as e:
        raise CustomException(e, sys)
    
def load_object(file_path):
    '''
    opens the saved pkl file in read by mode, and loads the file
    '''
    try:
        with open(file_path, "rb") as file_obj:
            return dill.load(file_obj)
        
    except Exception as e:
        raise CustomException(e, sys)