import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader as data
from keras.models import load_model
import streamlit as st

from pandas_datareader import data as pdr
import yfinance as yf
yf.pdr_override()
#y_symbols = ['TCS.NS', 'TATAPOWER.NS']
from datetime import datetime
startdate = '2010-01-01'
enddate  = '2022-12-20'
# data = pdr.get_data_yahoo('TCS', start=startdate,  end=enddate)
# data.head()

st.title('Stock Trend Precdiction')

user_input = st.text_input('Enter Stock Ticker', 'TCS')
data = pdr.get_data_yahoo(user_input, start=startdate,  end=enddate)

st.subheader('Data from 2010 - 2022')
st.write(data.describe())


st.subheader('Closing Price vs Time Chart ')
fig = plt.figure(figsize=(12,6))
plt.plot(data.Close)
st.pyplot(fig)

st.subheader('Closing Price vs Time Chart with 100MA')
ma100 = data.Close.rolling(100).mean()
fig = plt.figure(figsize=(12,6))
plt.plot(ma100)
plt.plot(data.Close)
st.pyplot(fig)

st.subheader('Closing Price vs Time Chart with 100MA and 200MA ')
ma100 = data.Close.rolling(100).mean()
ma200 = data.Close.rolling(100).mean()
fig = plt.figure(figsize=(12,6))
plt.plot(ma100)
plt.plot(ma200)
plt.plot(data.Close)
st.pyplot(fig)

#splitting  data into training and testing
data_training = pd.DataFrame(data['Close'][0:int(len(data)*0.70)])
data_testing = pd.DataFrame(data['Close'][int(len(data)*0.70): int(len(data))])

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))

data_training_array = scaler.fit_transform(data_training)


#loading the model
rom keras.optimizers import Adam

def custom_adam(lr=0.001):
    return Adam(lr=lr)

custom_objects = {'CustomAdam': custom_adam}
model = load_model('keras_model.h5', custom_objects=custom_objects)

#Testing part 

past_100_days = data_training.tail(100)
final_data = past_100_days.append(data_testing, ignore_index=True)
input_data = scaler.fit_transform(final_data)

x_test = []
y_test = []

for i in range(100, input_data.shape[0]):
    x_test.append(input_data[i-100: i])
    y_test.append(input_data[i, 0])

x_test, y_test = np.array(x_test), np.array(y_test)


# making predictions

y_predicted = model.predict(x_test)

scaler= scaler.scale_

scale_factor = 1/scaler[0]
y_predicted = y_predicted * scale_factor
y_test = y_test * scale_factor

#Final Vizualization

st.subheader('Prediction vs Origianl')
fig2 = plt.figure(figsize=(12,6))
plt.plot(y_test, 'b', label = 'Original Price')
plt.plot(y_predicted, 'r', label = 'Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig2)
