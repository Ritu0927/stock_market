
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pandas_datareader as data
import yfinance as yf
from keras.models import load_model
import streamlit as st

start = '2010-01-01'
end = '2019-12-01'

user_input=  st.text_input('Enter Stock Ticker','TSLA' )
df = yf.download(user_input, start=start, end=end)
print(df.head())
#Describing Data
st.subheader('Data from 2000- 2019')
st.write(df.describe())
 #VIsualizations
st.subheader('Closing Price vs Time Chart')
fig = plt.figure(figsize = (12,6))
plt.plot(df.Close)
st.pyplot(fig)

st.subheader('Closing Price vs Time CHart with 100Ma')
ma100= df.Close.rolling(100).mean()
print(ma100)
plt.plot(ma100)
plt.plot(df.Close)
st.pyplot(fig)

st.subheader('Closing Price vs Time chart with 100 Ma and 200Ma')
# ma100= df.Close.rolling(100).mean
ma200=df.Close.rolling(200).mean()
fig = plt.figure(figsize = (12,6))
plt.plot(ma100)
plt.plot(ma200)
plt.plot(df.Close)
st.pyplot(fig)



#Splitting data into training and testing
data_training_lstm = pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
data_testing_lstm = pd.DataFrame(df['Close'][int(len(df)*0.70):int(len(df))]) 


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range= (0,1))

data_training_array_lstm = scaler.fit_transform(data_training_lstm)

x_train_lstm= []
y_train_lstm= []

for i in range(100,data_training_array_lstm.shape[0]):
    x_train_lstm.append(data_training_array_lstm[i-100:i])
    y_train_lstm.append(data_training_array_lstm[i,0])
    
x_train_lstm, y_train_lstm= np.array(x_train_lstm),np.array(y_train_lstm)

from keras.layers import Dense, Dropout, LSTM
from keras.models import Sequential
model_LSTM = load_model("./lstm_model.h5")

# model_LSTM = Sequential()
# model_LSTM.add(LSTM(units = 50, activation = 'relu', return_sequences = True, input_shape = (x_train_lstm.shape[1], 1)))
# model_LSTM.add(Dropout(0.2))

# model_LSTM.add(LSTM(units = 60, activation = 'relu', return_sequences = True ))
# model_LSTM.add(Dropout(0.3))

# model_LSTM.add(LSTM(units = 80, activation = 'relu', return_sequences = True))
# model_LSTM.add(Dropout(0.4))

# model_LSTM.add(LSTM(units = 120, activation = 'relu'))
# model_LSTM.add(Dropout(0.5))

# model_LSTM.add(Dense (units = 1))

# model_LSTM.compile(optimizer= 'adam', loss= 'mean_squared_error')
# model_LSTM.fit(x_train_lstm, y_train_lstm, epochs = 50)

# data_testing_lstm.head()

# data_training_lstm.tail(100)

past_100_days= data_training_lstm.tail(100)


final_df_lstm = past_100_days._append(data_testing_lstm, ignore_index = True)

input_data_lstm= scaler.fit_transform(final_df_lstm)

x_test_lstm= []
y_test_lstm= []

for i in range(100, input_data_lstm.shape[0]):
        x_test_lstm.append(input_data_lstm[i-100:i])
        y_test_lstm.append(input_data_lstm[i,0])

x_test_lstm, y_test_lstm = np.array(x_test_lstm), np.array(y_test_lstm)

#making predictions
y_predicted_lstm= model_LSTM.predict(x_test_lstm)

scale_factor= 1/0.07280492
y_predicted_lstm = y_predicted_lstm*scale_factor
y_test_lstm = y_test_lstm*scale_factor

def calculate_accuracy(y_predicted_lstm, y_test_lstm):
   
    # Calculate the absolute error between predictions and actual values
    errors = np.abs(y_predicted_lstm - y_test_lstm)
    
    # Calculate the mean absolute error (MAE)
    mean_absolute_error = np.mean(errors)
    
    # Calculate the accuracy
    accuracy = 1 - (mean_absolute_error / np.mean(y_test_lstm))
    
    return accuracy

# Example usage:
y_predicted = np.array(y_predicted_lstm)
y_test = np.array(y_test_lstm)

accuracy_LSTM = calculate_accuracy(y_predicted_lstm, y_test_lstm)

 # Calculate the absolute error between predictions and actual values
errors_lstm = np.abs(y_predicted_lstm - y_test_lstm)
    
    # Calculate the mean absolute error (MAE)
mean_absolute_error_lstm = np.mean(errors_lstm)

def mean_squared_error(y_predicted_lstm, y_test_lstm):
   
    # Calculate the squared error
    squared_error = np.square(y_test_lstm - y_predicted_lstm)
    
    # Calculate the mean squared error
    mse_lstm = np.mean(squared_error)
    
    return mse_lstm
mse_lstm = mean_squared_error(y_test_lstm, y_predicted_lstm)


rmse_lstm = np.sqrt(mse_lstm)




fig1=plt.figure(figsize=(12,6))
st.header('LSTM PREDICTION')
plt.plot(y_test_lstm, 'b', label= 'Original Price')
plt.plot(y_predicted_lstm, 'r', label = 'Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
st.pyplot(fig1)

st.subheader('Metrics of LSTM MODEL')
st.subheader('Mean Squared Error of LSTM:')
mse_lstm
st.subheader('Accuracy of LSTM:')
accuracy_LSTM
st.subheader('Mean Absolute Error of LSTM:')
mean_absolute_error_lstm
st.subheader('Root Mean Squared Error of LSTM:')
rmse_lstm
print("Mean Absolute Error:",mean_absolute_error_lstm)
print("Root Mean Squared Error:", rmse_lstm)



from sklearn.ensemble import GradientBoostingRegressor

#Splitting data into training and testing
data_training_gb = pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
data_testing_gb = pd.DataFrame(df['Close'][int(len(df)*0.70):int(len(df))])

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range= (0,1))

data_training_array_gb = scaler.fit_transform(data_training_gb)

x_train_gb= []
y_train_gb= []

for i in range(100,data_training_array_gb.shape[0]):
    x_train_gb.append(data_training_array_gb[i-100:i])
    y_train_gb.append(data_training_array_gb[i,0])
    
x_train_gb, y_train_gb= np.array(x_train_gb),np.array(y_train_gb)

train_row,train_col,train_num=x_train_gb.shape

past_100_days= data_training_gb.tail(100)

final_df_gb = past_100_days._append(data_testing_gb, ignore_index = True)

input_data_gb= scaler.fit_transform(final_df_gb)

x_test_gb= []
y_test_gb= []

for i in range(100, input_data_gb.shape[0]):
        x_test_gb.append(input_data_gb[i-100:i])
        y_test_gb.append(input_data_gb[i,0])

x_test_gb, y_test_gb = np.array(x_test_gb), np.array(y_test_gb)

from sklearn.ensemble import GradientBoostingRegressor

# Reshape the input data to 2-dimensional arrays if necessary
x_train_gb = np.reshape(x_train_gb, (x_train_gb.shape[0], -1))

model_GB= GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=1, random_state=1)

model_GB.fit(x_train_gb, y_train_gb)

# Reshape the input data to 2-dimensional arrays if necessary
x_test_gb = np.reshape(x_test_gb, (x_test_gb.shape[0], -1))

# Use the model to predict sales for the test data
predictions = model_GB.predict(x_test_gb)

y_predicted_gb = model_GB.predict(x_test_gb)

def mean_squared_error(y_predicted_gb, y_test_gb):
   
    # Calculate the squared error
    squared_error_gb = np.square(y_test_gb - y_predicted_gb)
    
    # Calculate the mean squared error
    mse_gb = np.mean(squared_error_gb)
    
    return mse_gb
mse_gb = mean_squared_error(y_test_gb, y_predicted_gb)

    # Calculate the root mean squared error
rmse_gb = np.sqrt(mse_gb)

def calculate_accuracy(y_predicted_gb, y_test_gb):
   
    # Calculate the absolute error between predictions and actual values
    errors = np.abs(y_predicted_gb - y_test_gb)
    
    # Calculate the mean absolute error (MAE)
    mean_absolute_error = np.mean(errors)
    
    # Calculate the accuracy
    accuracy = 1 - (mean_absolute_error / np.mean(y_test_gb))
    
    return accuracy

# Example usage:
y_predicted = np.array(y_predicted)
y_test = np.array(y_test)

accuracy_gb = calculate_accuracy(y_predicted_gb, y_test_gb)

errors = np.abs(y_predicted_gb - y_test_gb)
# Calculate the mean absolute error (MAE)
mean_absolute_error_gb = np.mean(errors)



fig3= plt.figure(figsize=(12,6))
st.subheader('GRADIENT BOOSTING PREDICTION')
plt.plot(y_test_gb, 'b', label= 'Original Price')
plt.plot(y_predicted_gb, 'r', label = 'Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
st.pyplot(fig3)

st.subheader('Metrics of Gradient Boosting Model')
st.subheader('Mean Squared Error of GB:')
mse_gb
st.subheader('Accuracy of GB:')
accuracy_gb
st.subheader('Mean Absolute Error of GB:')
mean_absolute_error_gb
st.subheader('Root Mean Squared Error of GB:')
rmse_gb





