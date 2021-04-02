import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pandas_datareader as web
import datetime as dt
import logging

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM

def predict(symbol):

  logging.info("prediction")

  start = dt.datetime(2012,1,1)
  end = dt.datetime(2020,12,31)
  
  data = web.DataReader(symbol, 'yahoo', start, end)
  
  # prepare data
  scaler = MinMaxScaler(feature_range=(0,1))
  scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1,1))
  
  prediction_days = 60

  x_train = []
  y_train = []
  
  for x in range(prediction_days, len(scaled_data)):
    x_train.append(scaled_data[x-prediction_days:x, 0])
    y_train.append(scaled_data[x, 0])
    
  x_train, y_train = np.array(x_train), np.array(y_train)
  x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
  
  # build model
  model = Sequential()
  
  model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
  model.add(Dropout(0,2))
  model.add(LSTM(units=50, return_sequences=True))
  model.add(Dropout(0,2))
  model.add(LSTM(units=50, return_sequences=True))
  model.add(Dropout(0,2))
  model.add(Dense(units=1))
  
  model.compile(optimizer='adam', loss='mean_squared_error')
  model.fit(x_train, y_train, epochs=25, batch_size=32)
  
  # validate the model
  
  
  
