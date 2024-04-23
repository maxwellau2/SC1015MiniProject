# this code is meant to train models automatically
# ain't nobody got time to run cell by cell, i gotta do other stuff too

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datacleaningutils import datacleaning
from timeseriesutils import tsUtils
from datetime import datetime
import datatransformations as datatrans

trans = datatrans.datatransformation()

res = pd.read_csv("data/cleaned.csv")
# features = ['PM10', 'SO2', 'WSPM', 'hour', 'projection', 'PRES', 'month', 'year', 'day', 'O3', 'RAIN']
# columns_to_transform = ['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3', 'RAIN', 'WSPM']
# row = features + ['PM2.5']

columns = ['month', 'hour', 'PM10', 'SO2', 'NO2', 'CO', 'O3', 'DEWP', 'projection']
columns_to_transform = ['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3', 'RAIN', 'WSPM']
row = columns + ['PM2.5']
# print(res)
res['projection'] = res.apply(lambda row: trans.calculate_projection_vector(row, "SE"), axis=1)
df_yj, l = trans.yeojohnson_transform_columns(res,columns_to_transform)
df_sqrt = trans.sqrt_transform_column(res,columns_to_transform)
df_log = trans.log_transform_columns(res,columns_to_transform)

df_arr = [res,df_yj,df_sqrt,df_log] # include non-transformed dataset to compare
transtype = ['og', 'yj', 'sqrt', 'log']
base_filepath = "models/TimeSeriesModels/"
modeltype = ['gru', 'lstm']
for i in range(len(df_arr)):

    # df_arr[i]['projection'] = df_arr[i].apply(lambda row: trans.calculate_projection_vector(row, "SE"), axis=1)

    X_train = df_arr[i][row]
    ts = tsUtils()
    x_ts, y_ts = ts.generate_time_series(X_train, 10)

    print(x_ts.shape, y_ts.shape)

    # debugging to check shape, commented for good luck :)
    # import os
    # os.abort()

    x_ts_train = x_ts[:int(len(x_ts)*0.8)]
    y_ts_train = y_ts[:int(len(y_ts)*0.8)]

    x_ts_test = x_ts[int(len(x_ts)*0.8):]
    y_ts_test = y_ts[int(len(y_ts)*0.8):]

    # we abstracted the training code into this class method to simplify understanding
    model = ts.trainModel_LSTM(X_train_ts=x_ts_train, y_train_ts=y_ts_train, window_size=10, num_epochs=200)
    # train data metrics
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")  # Format: YYYYMMDD_HHMMSS
    folder_name = base_filepath + modeltype[1] + '/'  + transtype[i]
    model.save(f'{folder_name}/model_{timestamp}.h5')
    plot_train = ts.print_metrics_and_plot(model, x_ts_train, y_ts_train, f"{folder_name}/trainmetrics{timestamp}.txt", transtype[i])
    # test data metric
    plot_test = ts.print_metrics_and_plot(model, x_ts_test, y_ts_test, f"{folder_name}/testmetrics{timestamp}.txt", transtype[i])
    plot_train.show()
    plot_test.show()
    # model.save(f'model_{timestamp}.h5')