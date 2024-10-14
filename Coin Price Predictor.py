##
# QUESTIONS:
# what metric should i use? OCHL or volume? - currently using close price, but voume would also make sense
# also, minute by minute the open and close price changes, but i thought they were determined at the end of the day?
# should i really use all historic data for training? pre-2015 the data is v different to 2020-2024
# what kind of activation should i use?
# when you compile the NN, what kind of optimiser, loss?
# im currently getting the median close price but could do sum (bad) or mean. chose median to try and remove large jumps / outliers?
##

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import datetime
import time
import csv

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

input_csv = "btcusd_1-min_data.csv"

                             # y    m  d  h   m   s
startDate = datetime.datetime(2020, 1, 1, 00, 00, 00) # inclusive
endDate   = datetime.datetime(2024, 1, 1, 00, 00, 00) # exclusive

startDate = time.mktime(startDate.timetuple()) # turn them into unix
endDate = time.mktime(endDate.timetuple())

csvData = pd.read_csv(input_csv)

csvData["Datetime"] = pd.to_datetime(csvData["Timestamp"], unit="s")

timeSlice = csvData[(csvData["Timestamp"] >= startDate) & (csvData["Timestamp"] < endDate)]

timeSlice["Date"] = timeSlice["Datetime"].dt.date

dataGroup = timeSlice.groupby("Date")
closePriceGroup = dataGroup["Close"]            # change this to OCHL or volume - should probs generalise variable names because theyre just for close price

closePriceDataFrame = closePriceGroup.median()    # mean, median or sum? i have no idea

#print(closePriceDataFrame.values)

testDataFrame = closePriceDataFrame
trainingDataFrame = testDataFrame[::-1]  # for regression, the data is fed in backwards

trainingDataSet = trainingDataFrame.values
trainingDataSet = trainingDataSet.reshape(len(trainingDataSet),1)

#print(trainingDataSet)


# we are going to standardize our data set, either using z-score standardisation or min-max scaling (idk the difference)

def zScoreStandardisation(inputSet):
    mean = np.mean(inputSet, axis=0)
    std = np.std(inputSet, axis=0)
    standardizedSet = (inputSet - mean) / std
    return standardizedSet

standardizedTrainingData = zScoreStandardisation(trainingDataSet)


# THIS SECTION IS TAKEN FROM HERE: https://www.datacamp.com/tutorial/tutorial-for-recurrent-neural-network

lstmModel = Sequential()
lstmModel.add(LSTM(units=150, activation = "tanh", input_shape = (1,1)))      # more units = more learning / more training time
lstmModel.add(Dense(units=1))                                               # make sure to only output one unit - this is the single data point on the graph
lstmModel.compile(optimizer="RMSprop", loss="mean_squared_error")           # change these around?

##
# "A Sequential model is appropriate for a plain stack of layers where each layer has exactly one input tensor and one output tensor." - https://keras.io/guides/sequential_model/
# the LSTM layer processes the sequential market data - recognising patterns that occur within a certain sequence of time. these patterns are know as temporal dependancies
# the dense layer takes the output of the LSTM and returns a single predicted value
##

#lstmModel.summary()

lstmModel.fit(standardizedTrainingData, standardizedTrainingData, epochs = 500, batch_size = 32)

test_set = testDataFrame.values

testInput = test_set.reshape(len(test_set), 1)      # turn a wide dataframe into a tall data frame

# FIX - cant use the function for this because we use STD and mean a little later

mean = np.mean(testInput, axis=0)
std = np.std(testInput, axis=0)
standardizedTestInput = (testInput - mean) / std            # z score standardisation

#print(standardizedTestInput)

predictedPrice = lstmModel.predict(standardizedTestInput)       # put the standardised data into the model

originalPredictedPrice = (predictedPrice * std) + mean          # reverse the standardization of the predicted set to get the usable predicted price
                                                                # THIS USES THE STD AND MEAN NOT FROM THE STANDARDISED SET BUT FROM THE UNSTANDARDISED SET - IDK IF THIS CHANGES ANYTHING
dates = timeSlice["Datetime"].dt.date.unique()

plt.plot(dates[:len(originalPredictedPrice)], originalPredictedPrice, color = "red", label = "Predicted Price", linewidth = 0.75)
plt.plot(dates[:len(test_set)], test_set, color = "blue", label = "Actual Price", linewidth = 0.75)
plt.xlabel("Date")
plt.ylabel("Price (USD)")
plt.title("Predicted vs Actual Price")
plt.legend()
plt.show()

from sklearn.metrics import mean_squared_error

print(f"The mean squared error is: {mean_squared_error(test_set, originalPredictedPrice)}")

# save the predicted prices to a CSV file
outputDataFrame = pd.DataFrame({
    "Date": dates[:len(originalPredictedPrice)],
    "Predicted Close Price": originalPredictedPrice.flatten()
    })

outputCSV = "predictedPrices.csv"
outputDataFrame.to_csv(outputCSV, index=False)
print(f"Predicted prices saved to {outputCSV}")


# save the trained model
from keras.models import load_model

lstmModel.save("Trained LSTM Model.keras")
print("Model saved as Trained LSTM Model.keras")