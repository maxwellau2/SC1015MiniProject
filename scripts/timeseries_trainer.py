# this code is meant to train models automatically
# ain't nobody got time to run cell by cell, i gotta do other stuff too

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datacleaningutils import datacleaning
from miniproject_final.scripts.timeseriesutils import tsUtils
from datetime import datetime
import datatransformations as datatrans

trans = datatrans.datatransformation()

res = pd.read_csv("cleaned.csv")
columns = ['month', 'hour', 'PM10', 'SO2', 'NO2', 'CO', 'O3', 'DEWP', 'projection']
columns_to_transform = ['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3', 'RAIN', 'WSPM']
row = columns + ['PM2.5']
# print(res)
res['projection'] = res.apply(lambda row: trans.calculate_projection_vector(row, "SE"), axis=1)
res = trans.sqrt_transform_column(res, columns_to_transform)

X_train = res[row]
ts = tsUtils()
x_ts, y_ts = ts.generate_time_series_2(X_train, 10)
# # generate time series data
# x_ts = ts.generate_time_series(X_train)
# x_ts = x_ts[:-1]
# y_ts = ts.generate_target(res, row)

print(x_ts.shape, y_ts.shape)
# import os
# os.abort()

# split data into 80:20
x_ts_train = x_ts[:int(len(x_ts)*0.8)]
y_ts_train = y_ts[:int(len(y_ts)*0.8)]

x_ts_test = x_ts[int(len(x_ts)*0.8):]
y_ts_test = y_ts[int(len(y_ts)*0.8):]

# we abstracted the training code into this class method to simplify understanding
model = ts.trainModel_LSTM(X_train_ts=x_ts_train, y_train_ts=y_ts_train, window_size=10, num_epochs=100)
# train data metrics
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")  # Format: YYYYMMDD_HHMMSS
model.save(f'model_sqrt/model_{timestamp}.h5')
plot_train = ts.print_metrics_and_plot(model, x_ts_train, y_ts_train, f"model_sqrt_2_reducenodes/trainmetrics{timestamp}.txt")
# test data metric
plot_test = ts.print_metrics_and_plot(model, x_ts_test, y_ts_test, f"model_sqrt_2_reducenodes/testmetrics{timestamp}.txt")
plot_train.show()
plot_test.show()
# model.save(f'model_{timestamp}.h5')