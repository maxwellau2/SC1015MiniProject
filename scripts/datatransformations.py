import numpy as np
import pandas as pd

class datatransformation:
    def __init__(self):
        pass
    def log_transform_columns(self, dataframe, columns):
        transformed_df = dataframe.copy()
        for column in columns:
            transformed_df[column] = np.log1p(transformed_df[column])  # np.log1p to handle zero values
        return transformed_df
    
    def inverse_log_transform_column(self, column_data):
        return np.expm1(column_data)
    
    def sqrt_transform_column(self, dataframe, columns):
        transformed_df = dataframe.copy()
        for column in columns:
            transformed_df[column] = np.sqrt(transformed_df[column])
        return transformed_df
    
    def sqr_transform_column(self, column_data):
        return np.square(column_data)

    def calculate_projection_vector(self, row, reference_direction):
        # wind direction angles in degrees
        wind_direction_angles = {
            'N': 0.0, 'NNE': 22.5, 'NE': 45.0, 'ENE': 67.5,
            'E': 90.0, 'ESE': 112.5, 'SE': 135.0, 'SSE': 157.5,
            'S': 180.0, 'SSW': 202.5, 'SW': 225.0, 'WSW': 247.5,
            'W': 270.0, 'WNW': 292.5, 'NW': 315.0, 'NNW': 337.5
        }
        
        wind_speed = row['WSPM']
        wind_direction = row['wd']
        
        # map wind direction
        wind_direction_deg = wind_direction_angles[wind_direction]
        reference_direction_deg = wind_direction_angles[reference_direction]

        # dot product
        wind_speed_projection = wind_speed * np.cos(np.radians(wind_direction_deg - reference_direction_deg))
        
        return wind_speed_projection