# Basic Import
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import r2_score, mean_squared_error
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object
from dataclasses import dataclass
import sys
import os

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts', 'model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def evaluate_model(self, X_train, y_train, X_test, y_test, models):
        """ 
        Evaluates different regression models and returns their R2 score in a report.
        """
        model_report = {}
        for model_name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            test_r2_score = r2_score(y_test, y_pred)
            model_report[model_name] = test_r2_score
        return model_report

    def initate_model_training(self, train_array, test_array):
        try:
            logging.info('Splitting Dependent and Independent variables from train and test data')
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],  # Features
                train_array[:, -1],   # Target
                test_array[:, :-1],   # Features
                test_array[:, -1]     # Target
            )

            # Define models for training
            models = {
                "Linear Regression": LinearRegression(),
                "Ridge": Ridge(),
                "Lasso": Lasso(),
                "ElasticNet": ElasticNet(),
                "Decision Tree": DecisionTreeRegressor(),
                "Random Forest": RandomForestRegressor(),
                "Support Vector Regressor": SVR()
            }

            # Evaluate the models and get the report
            model_report = self.evaluate_model(X_train, y_train, X_test, y_test, models)
            print(model_report)
            print('\n====================================================================================\n')
            logging.info(f'Model Report : {model_report}')

            # Get the best model score from the report
            best_model_score = max(sorted(model_report.values()))

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            best_model = models[best_model_name]

            print(f'Best Model Found: {best_model_name}, R2 Score: {best_model_score}')
            print('\n====================================================================================\n')
            logging.info(f'Best Model Found: {best_model_name}, R2 Score: {best_model_score}')

            # Save the best model
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )
            logging.info('Best model saved successfully.')

        except Exception as e:
            logging.info('Exception occurred during model training')
            raise CustomException(e, sys)

