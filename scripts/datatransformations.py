import numpy as np
from scipy.stats import yeojohnson
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pandas as pd
import matplotlib.pylab as plt

class datatransformation:
    def __init__(self):
        self.lambda_ = {'PM2.5': 0.11821580362920309,
                'PM10': 0.18947730848393513,
                'SO2': -0.2404703077191753,
                'NO2': 0.22404372458954563,
                'CO': -0.10109070447505275,
                'O3': 0.1970739743881009,
                'RAIN': -36.75122189560069,
                'WSPM': -0.29073179806681426} # lambda value for yj transform
        pass

    def yeojohnson_transform_column(self, column_data):
        # Perform Ye-Johnson transformation
        transformed_data, lambda_ = yeojohnson(column_data)
        self.lambda_ = lambda_
        return transformed_data, lambda_

    def inverse_yeojohnson_transform_column(self, transformed_data, lambda_ = None):
        if lambda_ == None:
            lambda_ = self.lambda_
        # Perform inverse Ye-Johnson transformation
        if lambda_ == 0:
            return np.exp(transformed_data) - 1
        else:
            return np.power((transformed_data * lambda_ + 1), 1 / lambda_) - 1

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

    def yeojohnson_transform_columns(self, dataframe, columns):
        transformed_df = dataframe.copy()
        lambda_dict = {}
        for column in columns:
            transformed_data, lambda_ = self.yeojohnson_transform_column(transformed_df[column])
            transformed_df[column] = transformed_data
            lambda_dict[column] = lambda_
        return transformed_df, lambda_dict

    def inverse_yeojohnson_transform_columns(self, dataframe, columns, lambda_dict):
        inverse_transformed_df = dataframe.copy()
        for column in columns:
            inverse_transformed_df[column] = self.inverse_yeojohnson_transform_column(inverse_transformed_df[column], lambda_dict[column])
        return inverse_transformed_df

    def log_transform_columns(self, dataframe, columns):
        transformed_df = dataframe.copy()
        for column in columns:
            transformed_df[column] = np.log1p(transformed_df[column])
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
    
    def conduct_pca(self, df):
        # standardize data
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(df)

        # compute the covariance matrix
        cov_matrix = np.cov(scaled_data, rowvar=False)

        # compute eigenvectors and eigenvalues
        eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

        # select principal components
        # sort EVs
        sorted_indices = np.argsort(eigenvalues)[::-1]
        eigenvalues_sorted = eigenvalues[sorted_indices]
        eigenvectors_sorted = eigenvectors[:, sorted_indices]

        # choose number of principal components to retain
        explained_variance_ratio = np.cumsum(eigenvalues_sorted) / np.sum(eigenvalues_sorted)
        num_components = np.argmax(explained_variance_ratio >= 0.9) + 1

        # project data onto principal components
        pca = PCA(n_components=num_components)
        pca_result = pca.fit_transform(scaled_data)
        plt.plot(np.cumsum(pca.explained_variance_ratio_), marker='o')
        plt.xlabel('Number of Components')
        plt.ylabel('Cumulative Explained Variance')
        plt.title('Cumulative Explained Variance by Principal Components')
        plt.show()
        loadings = np.abs(pca.components_)
        loadings_df = pd.DataFrame(loadings, columns=df.columns)

        # identify the top features with the highest loadings on each principal component
        num_top_features = 8
        top_features = {}
        for i, component in enumerate(loadings_df.index):
            top_features[component] = loadings_df.iloc[i].nlargest(num_top_features).index.tolist()

        # # print top features for each principal component
        # for component, features in top_features.items():
        #     print(f"Principal Component {component + 1}:")
        #     print(features)

        # set a threshold for the number of appearances across components
        appearance_threshold = 2

        selected_variables = {}

        for pc, features in top_features.items():
            selected_variables[pc] = features

        # count appearances
        consistently_selected_variables = {}
        for features in top_features.values():
            for feature in features:
                if feature in consistently_selected_variables:
                    consistently_selected_variables[feature] += 1
                else:
                    consistently_selected_variables[feature] = 1

        # filter variables that appear in multiple principal components
        consistently_selected_variables = {feature: count for feature, count in consistently_selected_variables.items() if count >= appearance_threshold}
        sorted_consistently_selected_variables = dict(sorted(consistently_selected_variables.items(), key=lambda x: x[1], reverse=True))
        return sorted_consistently_selected_variables
        # for pc, features in selected_variables.items():
        #     print(f"Principal Component {pc}:")
        #     print(features)

        # print("\nConsistently Selected Variables:")
        # for feature, count in consistently_selected_variables.items():
        #     print(f"{feature}: {count} appearances")