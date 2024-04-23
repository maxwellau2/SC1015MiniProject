import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.impute import KNNImputer

class datacleaning:
    def __init__(self):
        pass
    def find_missing(self, df,column):
        missing_indices = df[df[column].isnull()].index

        # Identify windows of missing data
        missing_windows = []
        current_window = []
        for idx in missing_indices:
            if not current_window or current_window[-1] == idx - 1:
                current_window.append(idx)
            else:
                missing_windows.append(current_window)
                current_window = [idx]
        if current_window:
            missing_windows.append(current_window)

        if missing_windows != []:
            # Calculate the number of intervals spanned by each missing window
            intervals_spanned = [len(window) for window in missing_windows]

            # Print the results
            pt = 0
            print("Windows of missing data:")
            for window in missing_windows:
                # print(window)
                if len(window) == 1:
                    pt += 1
            print(f"Number of windows of missing data {len(missing_windows)}")
            # print("Number of intervals spanned by each missing window:", intervals_spanned)
            print("longest window of missing data:", len(intervals_spanned))
            print(f"Point missing data {pt}")

    # putting it all together
    def fill_missing_from_other_datasets(self, df, column_name, df_array, threshhold):
        # figure out if df is correlated enough to impute
        reference_dfs = []
        for i in df_array:
            corr = df[column_name].corr(i[column_name])
            if corr > threshhold and corr != 1:
                reference_dfs.append(i)
        # iterate thru dfs that meet requirement
        for i in range(len(df)):
            if pd.isna(df.at[i, column_name]):
                for reference_df in reference_dfs:
                    reference_value = reference_df.at[i, column_name]
                    if not pd.isna(reference_value):
                        df.at[i, column_name] = reference_value
                        break  # stop searching if a non-null value is found in any of the reference datasets
        return df

    def linear_interpolation_single_point_column(self, df, column_name):
        for i in range(1, len(df) - 1):
            if pd.isna(df.at[i, column_name]) and not pd.isna(df.at[i-1, column_name]) and not pd.isna(df.at[i+1, column_name]):
                x1 = df.at[i-1, column_name]
                x2 = df.at[i+1, column_name]
                df.at[i, column_name] = (x1 + x2) / 2
        return df
    
    def fill_missing(self, targets, columns = list):
        # then, impute missing values with correlated datasets
        for c in columns:
            for i in range(len(targets)):
                targets[i] = self.fill_missing_from_other_datasets(targets[i], c, targets, threshhold=0.8)
                # self.find_missing(targets[i], c)

            # first, perform single point linear interpolation
            for i in range(len(targets)):
                targets[i] = self.linear_interpolation_single_point_column(targets[i], c)
                # self.find_missing(targets[i], c)
        return targets

    def plot(df_arr, column):
        fig = go.Figure()
        for i in df_arr:
            fig.add_trace(go.Scatter(x=i.index, y=i[column], name=i['station'][0]))
        fig.show()

    def knn_imputer(self, df, n_neighbors=5):
        imputer = KNNImputer(n_neighbors=n_neighbors)
        df_filled = df.copy()
        filled_values = imputer.fit_transform(df_filled)
        df_filled = pd.DataFrame(filled_values, columns=df_filled.columns)
        return df_filled
    
    def clean_data(self, df_arr):
        for col in df_arr[0].columns:
            df_arr = self.fill_missing(df_arr, [col])
            # use knn imputer to fill in any remaining missing values
        for i in range(len(df_arr)):
            df_arr[i] = self.knn_imputer(df_arr[i])
        return df_arr
    
    def merge_dataframes_mean(self, dataframes):
        merged_df = pd.concat(dataframes)
        # calculate the mean across the rows
        mean_df = merged_df.groupby(level=0).mean()
        return mean_df