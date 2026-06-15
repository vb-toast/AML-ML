import os
import sys
from dataclasses import dataclass
from catboost import CatBoostClassifier
from sklearn.ensemble import(
    GradientBoostingClassifier,
    RandomForestClassifier
)
from sklearn.metrics import average_precision_score
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
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

            # Classifier models for binary AML detection — Is_Laundering: 0 (legitimate), 1 (laundering)
            models = { 
                "Random Forest": RandomForestClassifier(),
                "Decision Tree": DecisionTreeClassifier(),
                "Gradient Boosting": GradientBoostingClassifier(),
                "XGBClassifier": XGBClassifier(),
                "CatBoosting Classifier": CatBoostClassifier(verbose=False),
            } 
            
            params = {
                "Decision Tree": {
                    "criterion": ["gini", "entropy"], # controls how splis are evaluated, gini is fast and entropy works better on imbalanced data
                    "max_depth": [3, 5, 10, None], # ignores rare cases/overfitting
                    "min_samples_split": [2, 5, 10], # keep splits large
                    "class_weight": ["balanced"] 
                },
                "Random Forest": {
                    "n_estimators": [100, 200, 300], # stablizes predictions
                    "max_depth": [10, 20, None], 
                    "min_samples_split": [2, 5, 10],
                    "max_features": ["sqrt", "log2"], # because of spare signal features, reduces correlation based on 2 features to split on
                    "class_weight": ["balanced", "balanced_subsample"]
                },
                "Gradient Boosting": {
                    "learning_rate": [0.01, 0.05, 0.1], # lower the rate, the better the generalization, the longer it takes to load. This controls how much a tree correct vs the previous one
                    "n_estimators": [100, 200, 300], 
                    "max_depth": [3, 5, 7],
                    "subsample": [0.6, 0.8, 1.0], # adds stochasticity by training each tree on a random fraction of data, reduces overfitting
                    "min_samples_leaf": [1, 5, 10] # prevents splits that only capture a handful of laundering transactions
                },
                "XGBClassifier": {
                    "learning_rate": [0.01, 0.05, 0.1],
                    "n_estimators": [100, 200, 300],
                    "max_depth": [3, 5, 7],
                    "scale_pos_weight": [50, 100, 500, 999, 1704], # tells XGBoost how much upweight the positive class is. Searching across [50, 100, 500, 999] lets the grid find the right balance between recall and precision rather than hardcoding the extreme value
                    "subsample": [0.6, 0.8, 1.0],
                    "colsample_bytree": [0.6, 0.8, 1.0] # randomly samples features per tree
                },
                "CatBoosting Classifier": {
                    "depth": [4, 6, 8, 10],
                    "learning_rate": [0.01, 0.05, 0.1],
                    "iterations": [100, 200, 500], 
                    "auto_class_weights": ["Balanced"], # computes inverse frequency weights for each class
                    "l2_leaf_reg": [1, 3, 5, 7] # L2 regularization on lead values, higher values shrink predictions toward 0, useful when searching for rare values such as Is_Laundering
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

            if best_model_score < 0.1:
                raise CustomException("No model fits AUC-PR floor")
            
            logging.info("Best found model on both training and testing dataset")

            logging.info(f"Best model name: {best_model_name}, best model score: {best_model_score}")

            save_object(
                file_path = self.model_trainer_config.trained_model_file_path,
                obj = best_model
            ) 

            predicted_proba= best_model.predict_proba(x_test)[:, 1] # probability of being laundered

            auc_pr = average_precision_score(y_test, predicted_proba)
            return auc_pr

        except Exception as e:
            raise CustomException(e, sys)

