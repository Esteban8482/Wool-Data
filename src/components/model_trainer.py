import os
import sys
from dataclasses import dataclass

import numpy as np
from catboost import CatBoostRegressor, Pool, cv
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import RandomizedSearchCV, train_test_split

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_model

import mlflow
from mlflow import log_metric, log_param, log_artifact
import mlflow.sklearn

# Database Credentials
db_username = os.getenv('MLFLOW_DB_USERNAME')
db_password = os.getenv('MLFLOW_DB_PASSWORD')
db_host = os.getenv('MLFLOW_DB_HOST', 'localhost')
db_name = os.getenv('MLFLOW_DB_NAME')

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifact', 'model.pkl')
    mlflow_tracking_uri = f"mysql+pymysql://{db_username}:{db_password}@{db_host}/{db_name}"    
    
class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
        mlflow.set_tracking_uri(self.model_trainer_config.mlflow_tracking_uri)
        
    def initiate_model_trainer(self, train_array, test_array):
        with mlflow.start_run():
            try:
                logging.info("Split train and test input data")
                
                X_train, y_train, X_test, y_test = (
                    train_array[:, :-1],
                    train_array[:, -1],
                    test_array[:, :-1],
                    test_array[:, -1]
                )
                
                # Random Forest Implementation
                rf_hyperparameters = {
                    'n_estimators': [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)],
                    'max_depth': [int(x) for x in np.linspace(10, 110, num = 11)],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                }
                
                rf_model =RandomForestRegressor()
                rf_random_search = RandomizedSearchCV(
                    rf_model,
                    rf_hyperparameters,
                    n_iter = 10,
                    cv = 5,
                    verbose = 2,
                    random_state = 42,
                    n_jobs = -1
                )
                rf_random_search.fit(X_train, y_train)
                rf_best_model = rf_random_search.best_estimator_
                save_object(os.path.join('artifact', 'rf_model.pkl'), rf_best_model)
                
                # CatBoost Implementation
                cb_model = CatBoostRegressor(verbose = False)
                cb_params = {
                    'iterations' : [1000, 2000],
                    'learning_rate' : [0.01, 0.1, 0.3],
                    'depth' : [4, 6, 8, 10],
                    'l2_leaf_reg' : [1, 3, 5, 9]
                }
                cb_pool = Pool(X_train, y_train)
                cb_grid_search = cb_model.grid_search(
                    cb_params, 
                    cb_pool, 
                    cv = 5, 
                    stratified = False, 
                    verbose = False
                )
                cb_best_model = CatBoostRegressor(verbose = False, **cb_grid_search['params'])
                cb_best_model.fit(X_train, y_train)
                save_object(os.path.join('artifact', 'cb_model.pkl'), cb_best_model)
                
                # Evaluate the models
                rf_pred = rf_best_model.predict(X_test)
                rf_score = r2_score(y_test, rf_pred)
                logging.info(f"Random Forest R2 Score: {rf_score}")
                
                cb_pred = cb_best_model.predict(X_test)
                cb_score = r2_score(y_test, cb_pred)
                logging.info(f"CatBoost R2 Score: {cb_score}")
                
                # MLFlow Configuration
                mlflow.log_params(rf_random_search.best_params_)
                log_metric("Random Forest R2 Score", rf_score)
                mlflow.sklearn.log_model(rf_best_model, "RandomForest")
                
                mlflow.log_params(cb_grid_search['params'])
                log_metric("CatBoost R2 Score", cb_score)
                mlflow.sklearn.log_model(cb_best_model, "CatBoost")
                                
                save_object(os.path.join('artifact', 'rf_best_params.pkl'), rf_random_search.best_params_)
                save_object(os.path.join('artifact', 'cb_best_params.pkl'), cb_grid_search['params'])
                
                best_score = max(rf_score, cb_score)
                
                log_metric("Best R2 Score", best_score)
                
                if best_score < 0.6:
                    raise CustomException("Model performance is less than 0.6", sys)
                logging.info(f"Best R2 Score: {best_score}")
                
                return best_score
                
            except Exception as e:
                raise CustomException(e, sys)

            finally:
                mlflow.end_run()