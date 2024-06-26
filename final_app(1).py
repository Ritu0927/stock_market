
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
st.subheader('Mean Squared Error of LSTM: ')
mse_lstm
st.subheader('Accuracy of LSTM: ')
accuracy_LSTM
st.subheader('Mean Absolute Error of LSTM: ')
mean_absolute_error_lstm
st.subheader('Root Mean Squared Error of LSTM: ')
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
st.subheader('Mean Squared Error of GB: ')
mse_gb
st.subheader('Accuracy of GB: ')
accuracy_gb
st.subheader('Mean Absolute Error of GB: ')
mean_absolute_error_gb
st.subheader('Root Mean Squared Error of GB: ')
rmse_gb

# SVR..

from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

#Splitting data into training and testing
data_training_svm = pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
data_testing_svm = pd.DataFrame(df['Close'][int(len(df)*0.70):int(len(df))]) 

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range= (0,1))

data_training_array_svm = scaler.fit_transform(data_training_svm)

x_test_svm= []
y_test_svm= []

for i in range(100,data_training_array_svm.shape[0]):
    x_test_svm.append(data_training_array_svm[i-100:i])
    y_test_svm.append(data_training_array_svm[i,0])
    
x_test_svm, y_test_svm= np.array(x_test_svm),np.array(y_test_svm)

from joblib import load

# model_svm = load("./model_svm.joblib")

model_svm = SVR(kernel='linear')

x_test_svm = np.reshape(x_test_svm, (x_test_svm.shape[0], -1))
y_test_svm = np.reshape(y_test_svm, (y_test_svm.shape[0],))

model_svm.fit(x_test_svm, y_test_svm)

svm_predictions = model_svm.predict(x_test_svm)

fig_svm=plt.figure(figsize=(10, 6))
st.subheader('SVM Prediction')
plt.plot(y_test_svm, label='Actual')
plt.plot(svm_predictions, label='Predicted')
plt.title('Actual vs. Predicted Stock Prices')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
# plt.show()
st.pyplot(fig_svm)

def calculate_accuracy(svm_predictions, y_test_svm):
   
    # Calculate the absolute error between predictions and actual values
    errors = np.abs(svm_predictions - y_test_svm)
    
    # Calculate the mean absolute error (MAE)
    mean_absolute_error = np.mean(errors)
    
    # Calculate the accuracy
    accuracy = 1 - (mean_absolute_error / np.mean(y_test_svm))
    
    return accuracy

accuracy_svm = calculate_accuracy(svm_predictions, y_test_svm)
print("Accuracy:", accuracy_svm)

errors = np.abs(svm_predictions - y_test_svm)
# Calculate the mean absolute error (MAE)
mean_absolute_error_svm = np.mean(errors)
print("Mean Absolute Error:",mean_absolute_error_svm)

def mean_squared_error(svm_predictions, y_test_svm):
   
    # Calculate the squared error
    squared_error_svm = np.square(y_test_svm - svm_predictions)
    
    # Calculate the mean squared error
    mse_svm = np.mean(squared_error_svm)
    
    return mse_svm
mse_svm = mean_squared_error(y_test_svm, svm_predictions)
print("Mean Squared Error:", mse_svm)

   
    # Calculate the root mean squared error
rmse_svm = np.sqrt(mse_svm)
    
print("Root Mean Squared Error:"+ str(rmse_svm))

st.subheader('Metrics of SVM Model')
st.subheader('Mean Squared Error of SVM: ')
mse_svm
st.subheader('Accuracy of SVM: ')
accuracy_svm
st.subheader('Mean Absolute Error of SVM: ')
mean_absolute_error_svm
st.subheader('Root Mean Squared Error of SVM: ')
rmse_svm


# GRU Model

from keras.models import Sequential
from keras.layers import GRU, Dense
import matplotlib.pyplot as plt

model_GRU = load_model("./GRU_model.h5")

# Preprocessing data
def preprocess_data(data):
    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1,1))
    return scaled_data, scaler

def create_sequences(data, seq_length):
    x, y = [], []
    for i in range(len(data) - seq_length):
        x.append(data[i:(i + seq_length)])
        y.append(data[i + seq_length])
    return np.array(x), np.array(y)

scaled_data, scaler = preprocess_data(df)
seq_length = 30  # Sequence length for training    
    # Creating sequences and labels
x, y = create_sequences(scaled_data, seq_length)

 # Splitting data into training and testing sets
split = int(0.8 * len(x))
x_train, y_train = x[:split], y[:split]
x_test, y_test = x[split:], y[split:]

#  Reshaping data for GRU input
x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))
x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))


# Making predictions
GRU_predictions = model_GRU.predict(x_test)
GRU_predictions = scaler.inverse_transform(GRU_predictions)
y_test = scaler.inverse_transform(y_test.reshape(-1,1))

# Plotting results
fig_GRU=plt.figure(figsize=(12, 6))
st.subheader('GRU Prediction')
plt.plot(GRU_predictions, label='Predicted')
plt.plot(y_test, label='True')
plt.legend()
plt.show()
st.pyplot(fig_GRU)


def calculate_accuracy(GRU_predictions, y_test):
   
    # Calculate the absolute error between predictions and actual values
    errors = np.abs(GRU_predictions - y_test)
    
    # Calculate the mean absolute error (MAE)
    mean_absolute_error = np.mean(errors)
    
    # Calculate the accuracy
    accuracy = 1 - (mean_absolute_error / np.mean(y_test))
    
    return accuracy

accuracy_gru = calculate_accuracy(GRU_predictions, y_test)
print("Accuracy:", accuracy_gru)

from sklearn.metrics import mean_absolute_error, mean_squared_error

# Evaluate the model
def evaluate_model(y_test, GRU_predictions):
    mae_gru = mean_absolute_error(y_test, GRU_predictions)
    mse_gru = mean_squared_error(y_test, GRU_predictions)
    rmse_gru = np.sqrt(mse_gru)
    return mae_gru, mse_gru, rmse_gru

# Calculate accuracy metrics
mae_gru, mse_gru, rmse_gru = evaluate_model(y_test, GRU_predictions)
print("Mean Absolute Error (MAE):", mae_gru)
print("Mean Squared Error (MSE):", mse_gru)
print("Root Mean Squared Error (RMSE):", rmse_gru)

st.subheader('Metrics of GRU Model')
st.subheader('Mean Squared Error of GRU: ' )
mse_gru
st.subheader('Accuracy of GRU: ')
accuracy_gru
st.subheader('Mean Absolute Error of GRU: ')
mae_gru
st.subheader('Root Mean Squared Error of GRU: ')
rmse_gru


# Data
chart_data = pd.DataFrame(
   {
        "MODELS": ['LSTM','GB', 'SVM', 'GRU'],
         "Accuracy": [accuracy_LSTM , accuracy_gb, accuracy_svm, accuracy_gru],
           "MSE": [mse_lstm, mse_gb, mse_svm ,mse_gru], 
       "RMSE": [rmse_lstm, rmse_gb, rmse_svm, rmse_gru],
       "MAE":[mean_absolute_error_lstm , mean_absolute_error_gb , mean_absolute_error_svm , mae_gru],
   }
)
#Accuracy
st.subheader('Accuracy Comparison:')
st.bar_chart(chart_data, x="MODELS", y="Accuracy")

#MSE
st.subheader('MSE Comparison:')
st.bar_chart(chart_data, x="MODELS", y="MSE")

#RMSE
st.subheader('RMSE Comparison:')
st.bar_chart(chart_data, x="MODELS", y="RMSE")

#MAE
st.subheader('MAE Comparison:')
st.bar_chart(chart_data, x="MODELS", y="MAE")

# # Plot
# fig_plot=plt.figure(figsize=(10, 6))

# #MSE
# fig_mse, ax_mse = plt.subplots()
# st.subheader('Mean Squared Error Comparison:')
# ax_mse.bar(models, mse)
# # Adding labels to x-axis indexes
# ax_mse.set_xticks(range(len(models)))
# ax_mse.set_xticklabels(models)
# # Display the plot
# st.pyplot(fig_mse)

# #RMSE
# fig_rmse, ax_rmse = plt.subplots()
# st.subheader('Root Mean Squared Error Comparison:')
# ax_rmse.bar(models, rmse)
# # Adding labels to x-axis indexes
# ax_rmse.set_xticks(range(len(models)))
# ax_rmse.set_xticklabels(models)
# # Display the plot
# st.pyplot(fig_rmse)

# #MAE
# #MSE
# fig_mae, ax_mae = plt.subplots()
# st.subheader('Mean Absolute Error Comparison:')
# ax_mae.bar(models, mae)
# # Adding labels to x-axis indexes
# ax_mae.set_xticks(range(len(models)))
# ax_mae.set_xticklabels(models)
# # Display the plot
# st.pyplot(fig_mae)

# #ACCURACY
# #MSE
# fig_accuracy, ax_accuracy = plt.subplots()
# st.subheader('Accuracy Comparison:')
# # st.bar_chart(np.array(accuracy))
# ax_accuracy.bar(models, accuracy)
# # Adding labels to x-axis indexes
# ax_accuracy.set_xticks(range(len(models)))
# ax_accuracy.set_xticklabels(models)
# # Display the plot
# st.pyplot(fig_accuracy)

# chart_data = pd.DataFrame(
#    {
#        "col1": [accuracy_LSTM , accuracy_gb, accuracy_svm, accuracy_gru],
#        "col2": ['LSTM','Gradient Boosting', 'SVM', 'GRU'],
       
#    }
# )



# # Accuracy
# fig_accuracy=plt.subplot(2, 2, 2)
# st.subheader('Accuracy Comparision:')
# st.bar_chart(np.array(accuracy))

# # MAE
# fig_mae=plt.subplot(2, 2, 3)
# st.subheader('Mean Absolute Error Comparision:')
# st.bar_chart(np.array(mae))

# # RMSE
# fig_rmse=plt.subplot(2, 2, 4)
# st.subheader('Root Mean Squared Error Comparision:')
# st.bar_chart(np.array(rmse))





