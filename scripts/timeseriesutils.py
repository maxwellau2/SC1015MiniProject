import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error, r2_score
import plotly.graph_objects as go
from .datatransformations import datatransformation

class tsUtils:
    def __init__(self):
        pass

    def generate_time_series(self, res, window_size=10):
        time_series_data = []
        targets = []

        # Iterate over the DataFrame to create sequences
        for i in range(len(res) - window_size):
            window_start = i
            window_end = i + window_size
            target_index = window_end

            # Extract input features (window) and target value
            window_data = res.iloc[window_start:window_end]
            target_data = res.iloc[target_index]

            # Append the window data and target value
            time_series_data.append(window_data.values)
            targets.append(target_data.values)

        # Convert the lists to numpy arrays
        time_series_data = np.array(time_series_data)
        targets = np.array(targets)
        return time_series_data, targets
    
    def trainModel_GRU(self, X_train_ts, y_train_ts, window_size=10, num_epochs=100):
        input_dim = X_train_ts[0].shape[1]  # Number of input features
        output_dim = y_train_ts.shape[1]
        hidden_dim = 4*input_dim
        sequence_length = window_size
        batch_size = 32

        model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(sequence_length, input_dim)),
        tf.keras.layers.GRU(hidden_dim, return_sequences=True),  # GRU layer with return sequences for subsequent layers
        tf.keras.layers.Dropout(0.2),  # Dropout layer with dropout rate of 0.2
        tf.keras.layers.GRU(hidden_dim, return_sequences=True),  # Add another GRU layer
        tf.keras.layers.Dropout(0.2),  # Dropout layer with dropout rate of 0.2
        tf.keras.layers.GRU(hidden_dim, return_sequences=True),  # Add another GRU layer
        tf.keras.layers.Dropout(0.2),  # Dropout layer with dropout rate of 0.2
        tf.keras.layers.GRU(hidden_dim),  # Add another GRU layer
        tf.keras.layers.Dropout(0.2),  # Dropout layer with dropout rate of 0.2
        tf.keras.layers.Dense(hidden_dim, activation='relu'),  # Add a dense layer with ReLU activation
        tf.keras.layers.Dense(output_dim)  # Output layer
        ])
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5, min_lr=2.5000e-04)

        # Compile the model
        rate = 0.001  # You can adjust this value as needed
        optimizer = tf.keras.optimizers.Adam(learning_rate=rate)
        model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
        # Train the model
        history = model.fit(X_train_ts, y_train_ts, epochs=num_epochs, batch_size=batch_size, 
                            validation_split=0.2, callbacks=[early_stopping, lr_scheduler])
        return model

    def trainModel_LSTM(self, X_train_ts, y_train_ts, window_size=10, num_epochs=100):
        input_dim = X_train_ts[0].shape[1]  # Number of input features
        output_dim = y_train_ts.shape[1]
        hidden_dim= 4*input_dim
        sequence_length = window_size
        batch_size = 32

        model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(sequence_length, input_dim)),
        tf.keras.layers.LSTM(hidden_dim, return_sequences=True),  # GRU layer with return sequences for subsequent layers
        tf.keras.layers.Dropout(0.2),  # dropout rate of 0.2
        tf.keras.layers.LSTM(hidden_dim, return_sequences=True),  # another GRU layer
        tf.keras.layers.Dropout(0.2),  
        tf.keras.layers.LSTM(hidden_dim, return_sequences=True),  
        tf.keras.layers.Dropout(0.2),  
        tf.keras.layers.LSTM(hidden_dim),  
        tf.keras.layers.Dropout(0.2),  
        tf.keras.layers.Dense(hidden_dim, activation='relu'),  # dense layer with relu activation
        tf.keras.layers.Dense(output_dim)  # Output layer
        ])
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5, min_lr=2.5000e-04)

        # Compile the model
        rate = 0.001  # You can adjust this value as needed
        optimizer = tf.keras.optimizers.Adam(learning_rate=rate)
        model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
        # Train the model
        history = model.fit(X_train_ts, y_train_ts, epochs=num_epochs, batch_size=batch_size, 
                            validation_split=0.2, callbacks=[early_stopping, lr_scheduler])
        return model
    
    def retrain_model(self, model, X_train_ts, y_train_ts, num_epochs=100):
        batch_size = 32
        # Compile the model with the same optimizer, loss function, and metrics as before
        rate = 0.001  # You can adjust this value as needed
        optimizer = tf.keras.optimizers.Adam(learning_rate=rate)
        model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='mae', patience=10, restore_best_weights=True)
        # lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5, min_lr=2.5000e-04)
        # Train the model on the new data
        history = model.fit(X_train_ts, y_train_ts, epochs=num_epochs, batch_size=batch_size, callbacks=[early_stopping])
        
        return model

    
    def print_metrics_and_plot(self,model, x_ts, y_ts, file_name, transformation = None, output = True):
        trans = datatransformation()
        predicted = model.predict(x_ts)
        # extract PM2.5 values

        predicted_pm25 = np.array(predicted[:, -1])
        actual_pm25 = np.array(y_ts[:, -1])
        
        # evaluation metrics for PM2.5
        mae = mean_absolute_error(actual_pm25, predicted_pm25)
        mse = mean_squared_error(actual_pm25, predicted_pm25)
        mape = mean_absolute_percentage_error(actual_pm25, predicted_pm25)
        r2 = r2_score(actual_pm25, predicted_pm25)

        if output:
            # save evaluation metrics for PM2.5
            with open(file_name, "a") as f:
                print("Training Data Metrics for PM2.5:", file=f)
                print("Mean Absolute Error (MAE):", mae, file=f)
                print("Mean Squared Error (MSE):", mse, file=f)
                print("Mean Absolute Percent Error (MAPE):", mape, file=f)
                print("Explainable Variance (R^2):", r2, file=f)
        else:
            print("Training Data Metrics for PM2.5:")
            print("Mean Absolute Error (MAE):", mae )
            print("Mean Squared Error (MSE):", mse)
            print("Mean Absolute Percent Error (MAPE):", mape)
            print("Explainable Variance (R^2):", r2)
            
        # plot the predicted values for PM2.5
        print(predicted_pm25.shape, actual_pm25.shape)
        if transformation == "log":
            predicted_pm25 = trans.inverse_log_transform_column(predicted_pm25.flatten())
            actual_pm25 = trans.inverse_log_transform_column(actual_pm25.flatten())
        if transformation == "yj":
            predicted_pm25 = trans.inverse_yeojohnson_transform_column(predicted_pm25.flatten(), column="PM2.5")
            actual_pm25 = trans.inverse_yeojohnson_transform_column(actual_pm25.flatten(), column="PM2.5")
        if transformation == "sqrt":
            predicted_pm25 = trans.sqr_transform_column(predicted_pm25.flatten())
            actual_pm25 = trans.sqr_transform_column(actual_pm25.flatten())
        else:
            predicted_pm25 = predicted_pm25.flatten()
            actual_pm25 = actual_pm25.flatten()
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=predicted_pm25, name="Predicted PM2.5"))
        fig.add_trace(go.Scatter(y=actual_pm25, name="Actual PM2.5"))
        fig.update_layout(title=f"PM2.5 Output to {file_name}")
        return fig