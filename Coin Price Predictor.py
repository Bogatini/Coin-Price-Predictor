import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import datetime
import time
import csv

input_csv = "btcusd_1-min_data.csv"

                             # y    m  d  h   m   s
startDate = datetime.datetime(2020, 1, 1, 00, 00, 00) # inclusive
endDate =   datetime.datetime(2024, 1, 1, 00, 00, 00) # exclusive

startDate = time.mktime(startDate.timetuple()) # turn them into unix
endDate = int(time.mktime(endDate.timetuple()))

csvData = pd.read_csv(input_csv)

csvData["Datetime"] = pd.to_datetime(csvData["Timestamp"], unit="s")

timeSlice = csvData[(csvData["Timestamp"] >= startDate) & (csvData["Timestamp"] < endDate)]

timeSlice["Date"] = timeSlice["Datetime"].dt.date

dataGroup = timeSlice.groupby("Date")
closePriceGroup = dataGroup["Close"]

closePriceDataFrame = closePriceGroup.median()    # mean, median or sum? i have no idea

print(closePriceDataFrame.values)

testDataFrame = closePriceDataFrame
trainingDataFrame = realDataFrame[::-1]  # for regression, the data is fed in backwards

trainingDataSet = trainingDataFrame.values
trainingDataSet = trainingDataSet.reshape(len(trainingDataSet),1)

print(trainingDataSet)



# we are going to standardize our data set, either using z-score standardisation or min-max scaling

mean = np.mean(trainingDataSet, axis=0)
std = np.std(trainingDataSet, axis=0)
standardizedTrainingData = (trainingDataSet - mean) / std

# THIS SECTION IS TAKEN FROM HERE: https://www.datacamp.com/tutorial/tutorial-for-recurrent-neural-network




from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

lstmModel = Sequential()
lstmModel.add(LSTM(units=125, activation = "tanh", input_shape = (1,1)))
lstmModel.add(Dense(units=1))
lstmModel.compile(optimizer="RMSprop", loss="mean_squared_error")

#lstmModel.summary()



lstmModel.fit(standardizedTrainingData, standardizedTrainingData, epochs = 50, batch_size = 32)


test_set = testDataFrame.values
input = test_set.reshape(len(test_set), 1)



