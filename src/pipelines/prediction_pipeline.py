import sys
import os
import pandas as pd
from src.exception import CustomException
from src.logger import logging
from src.utils import load_object

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            preprocessor_path = os.path.join('artifacts', 'preprocessor.pkl')
            model_path = os.path.join('artifacts', 'model.pkl')

            preprocessor = load_object(preprocessor_path)
            model = load_object(model_path)

            # Scale the features
            data_scaled = preprocessor.transform(features)

            # Make predictions
            pred = model.predict(data_scaled)
            return pred

        except Exception as e:
            logging.info("Exception occurred in prediction")
            raise CustomException(e, sys)

class CustomData:
    def __init__(self, RH, Temperature, Rain, WindSpeed, FFMC, DMC, ISI, Classes, Region, DC):  # Added DC here
        self.RH = RH
        self.Temperature = Temperature
        self.Rain = Rain
        self.WindSpeed = WindSpeed
        self.FFMC = FFMC  # Fine Fuel Moisture Code
        self.DMC = DMC    # Duff Moisture Code
        self.ISI = ISI    # Initial Spread Index
        self.Classes = Classes
        self.Region = Region
        self.DC = DC      # Added DC here

    def get_data_as_dataframe(self):
        try:
            # Create a dictionary of the inputs
            custom_data_input_dict = {
                'RH': [self.RH],
                'Temperature': [self.Temperature],
                'Rain': [self.Rain],
                'Ws': [self.WindSpeed],
                'FFMC': [self.FFMC],
                'DMC': [self.DMC],
                'ISI': [self.ISI],
                'Classes': [self.Classes],
                'Region': [self.Region],
                'DC': [self.DC]  # Added DC here
            }

            # Convert dictionary to DataFrame
            df = pd.DataFrame(custom_data_input_dict)
            logging.info('Dataframe gathered')
            return df

        except Exception as e:
            logging.info('Exception occurred in prediction pipeline')
            raise CustomException(e, sys)




