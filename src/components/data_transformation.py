from sklearn.impute import SimpleImputer  # Handling Missing Values
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

import sys, os
from dataclasses import dataclass
import pandas as pd
import numpy as np

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object


## Data Transformation config
@dataclass
class DataTransformationconfig:
    preprocessor_obj_file_path = os.path.join('artifacts', 'preprocessor.pkl')


## Data Transformation class
class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationconfig()

    def get_data_transformation_object(self):
        """
        This function returns a preprocessor object for the data transformation pipeline.
        It handles missing values, encoding, and scaling.
        """
        try:
            logging.info('Data Transformation initiated')

            # Define which columns should be ordinal-encoded and which should be scaled
            numerical_columns = ['Temperature', 'RH', 'Ws', 'Rain', 'FFMC', 'DMC', 'DC', 'ISI']  # example numerical columns
            categorical_columns = ['Classes']  
            
            # Numerical pipeline (Imputation + Scaling)
            numerical_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')),  # Handle missing values with median
                ('scaler', StandardScaler())                    # Standardize numerical features
            ])

            # Categorical pipeline (Imputation + OneHotEncoding)
            categorical_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),  # Handle missing values with most frequent
                ('onehot', OneHotEncoder(handle_unknown='ignore'))     # OneHotEncode categorical variables
            ])

            # Combine numerical and categorical pipelines
            preprocessor = ColumnTransformer(transformers=[
                ('num_pipeline', numerical_pipeline, numerical_columns),
                ('cat_pipeline', categorical_pipeline, categorical_columns)
            ])

            logging.info('Pipeline Initiated')
            return preprocessor

        except Exception as e:
            logging.error("Error in Data Transformation")
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        """
        This function reads the train and test data, applies the preprocessing, and returns
        the transformed arrays along with the path to the preprocessor object file.
        """
        try:
            # Reading train and test data
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info('Read train and test data completed')
            logging.info(f'Train Dataframe Head : \n{train_df.head().to_string()}')
            logging.info(f'Test Dataframe Head  : \n{test_df.head().to_string()}')

            logging.info('Obtaining preprocessing object')

            preprocessing_obj = self.get_data_transformation_object()

            target_column_name = 'FWI'  # Target column (dependent feature)
            drop_columns = [target_column_name]  # Columns to drop from the features

            # Separate the features (X) and target (y) for both training and testing datasets
            input_feature_train_df = train_df.drop(columns=drop_columns, axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=drop_columns, axis=1)
            target_feature_test_df = test_df[target_column_name]

            logging.info("Applying preprocessing object on training and testing datasets.")

            # Apply transformations on training and testing datasets
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            # Combine transformed input features with target columns
            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            # Save the preprocessing object
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            logging.info('Preprocessor pickle is created and saved')

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )

        except Exception as e:
            logging.error("Exception occurred in initiate_data_transformation")
            raise CustomException(e, sys)
