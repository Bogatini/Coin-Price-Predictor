##
# QUESTIONS:
#
# should i really use all historic data for training? pre-2015 the data is v different to 2020-2024
# compare pre and post 2020 data
# +++ post trump election, it spiked like 40%
#
# what kind of activation should i use?
# investigte and write about this in diss
#
# when you compile the NN, what kind of optimiser, loss?
# ``
##

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"     # stops tensorflow warning messages about floating point precision (does not matter for our implementation)

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import datetime
import time
import csv
import kagglehub

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Input
from keras.layers import Dropout

save = False # determines if the model and predicted data is saved

showProgress = 0    # 0 = dont show trainging  1 = show bar  2 = just show epoch #
showGraph = True
epochs = 50
LSTMUnits = 100

metric = "Close" #OHCL or volume

                             # y    m  d  h   m   s
startDate = datetime.datetime(1999, 1, 1, 00, 00, 00) # inclusive
#endDate   = datetime.datetime(2021, 1, 1, 00, 00, 00) # exclusive
endDate = datetime.datetime.now()

startDateUnix = time.mktime(startDate.timetuple()) # turn them into unix
endDateUnix = time.mktime(endDate.timetuple())

# download latest version of the data set
path = kagglehub.dataset_download("mczielinski/bitcoin-historical-data")

print("Path to dataset files:", path)

csvData = pd.read_csv(path + "\\btcusd_1-min_data.csv", header = 0)

csvData["Datetime"] = pd.to_datetime(csvData["Timestamp"], unit = "s")

timeSlice = csvData[(csvData["Timestamp"] >= startDateUnix) & (csvData["Timestamp"] < endDateUnix)]

# pandas doesn't like set values onto copies of slices of dataframes, so we should use .loc() instead
# the colon denotes a null argument
timeSlice = timeSlice.copy()                                    # honestly no idea why this changes anything but it gets rid of the warning
timeSlice.loc[:, "Date"] = timeSlice["Datetime"].dt.date

# group the data into daily averages

dataGroup = timeSlice.groupby("Date")
closePriceGroup = dataGroup[metric]            # change this to OCHL or volume - should probs generalise variable names because theyre just for close price

closePriceDataFrame = closePriceGroup.median()    # mean, median or sum? i have no idea

#print(closePriceDataFrame.values)

testDataFrame = closePriceDataFrame
trainingDataFrame = testDataFrame[::-1]  # for regression, the data is fed in backwards

trainingDataSet = trainingDataFrame.values
trainingDataSet = trainingDataSet.reshape(len(trainingDataSet), 1)

# we are going to standardize our data set, either using z-score standardisation or min-max scaling (idk the difference)

def zScoreStandardisation(inputSet):
    mean = np.mean(inputSet, axis = 0)
    std = np.std(inputSet, axis = 0)
    standardizedSet = (inputSet - mean) / std
    return standardizedSet

standardisedTrainingData = zScoreStandardisation(trainingDataSet)

# THIS SECTION IS TAKEN FROM HERE: https://www.datacamp.com/tutorial/tutorial-for-recurrent-neural-network

lstmModel = Sequential()
lstmModel.add(Input((1,1)))                     # when using sequential models, an input layer is needed to take single values. these are then passed to the LSTM layer
lstmModel.add(LSTM(units = LSTMUnits, activation = "tanh"))    # more units = more learning / more training time
lstmModel.add(Dropout(0.2))
lstmModel.add(Dense(units = 1))                                               # make sure to only output one unit - this is the single data point on the graph
lstmModel.compile(optimizer = "RMSprop", loss = "mean_squared_error")         # change these around?

##
# "A Sequential model is appropriate for a plain stack of layers where each layer has exactly one input tensor and one output tensor." - https://keras.io/guides/sequential_model/
# the LSTM layer processes the sequential market data - recognising patterns that occur within a certain sequence of time. these patterns are know as temporal dependancies
# the dense layer takes the output of the LSTM and returns a single predicted value
##

#print("\n")
#lstmModel.summary()

inputData = standardisedTrainingData[:-1]   # all items but the very last one
inputData = inputData.reshape(len(inputData), 1, 1)     # the LSTM model needs a 3D vector - standardisedTrainingData is 1D and inputData is 3D     # the second dimension determines the # of time
targetData = standardisedTrainingData[1:]   # all item but the first one - this is what we're predicting                                # steps used in prediction, this may need to be
                                                                                                                                        #  increased later?

# we are constructing a supervised learning system here. To do this we have to change the lengths
# of out input and output data. We lop off the last item in standardisedTrainingData for the inputData
# and the first item in standardisedTrainingData for targetData. This makes their lengths the same, which
# allows the model to learn the relationship between items. Specifically the model maps an item to its
# subsequent item. For example, standardisedTrainingData[1] predicts standardisedTrainingData[2] which
# predicts standardisedTrainingData[3] and so on. (the last value is not used as it has no corresponding next item)

# we are creating a one-to-one mapping of previous values to future predicitons !!! IMPORTANT

# train the model using the new 3D vector inputData (returns the training data such as loss)
history = lstmModel.fit(inputData, targetData, epochs = epochs, batch_size = 32, verbose = showProgress)


actualPrices = testDataFrame.values

testInput = actualPrices.reshape(len(actualPrices), 1)      # turn a wide vector into a tall data frame

standardizedTestInput = zScoreStandardisation(testInput)            # z score standardisation

#print(standardizedTestInput)

predictedPrice = lstmModel.predict(standardizedTestInput, verbose = showProgress)       # put the standardised data into the model

mean = np.mean(testInput, axis = 0)                             # reverse the standardization of the predicted set to get the usable predicted price
std = np.std(testInput, axis = 0)                               # THIS USES THE STD AND MEAN NOT FROM THE STANDARDISED SET BUT FROM THE UNSTANDARDISED SET - THIS IS CORRECT

finalPredictedPrice = (predictedPrice * std) + mean

# loss graph
plt.plot(history.history["loss"], label= "Training Loss", color="orange")
plt.title("Model Loss Over Time")
plt.ylabel("Loss")
plt.xlabel("Epochs")
plt.legend()
plt.grid()

if showGraph:
    plt.show()

# price graph
dates = timeSlice["Datetime"].dt.date.unique()

plt.plot(dates[:len(finalPredictedPrice)], finalPredictedPrice, color = "red", label = "Predicted" + metric + " Price", linewidth = 0.75)
plt.plot(dates[:len(actualPrices)], actualPrices, color = "blue", label = "Actual" + metric + " Price", linewidth = 0.75)
plt.xticks(rotation=45)
plt.xlabel("Date")
plt.ylabel("Price (USD)")
plt.title("Predicted vs Actual Price")
plt.legend()

if showGraph:
    plt.show()


# this is crap and optional but it checks if the gradients are predicted correctly (they are 100% of the time)
# this means using gradient to decide whether to buy/sell could be good

actualGradient = np.diff(actualPrices)
predictedGradient = np.diff(finalPredictedPrice)

for a in actualGradient:
    if a >0:
        a=1
    else:
        a=-1

for a in predictedGradient:
    if a >0:
        a=1
    else:
        a=-1

for x in range(len(dates)-1):
    if actualGradient[x] != predictedGradient[x]:
        print("Gradients do not match!")
## end of optional gradient stuff


from sklearn.metrics import mean_squared_error

print(f"The mean squared error is: {mean_squared_error(actualPrices, finalPredictedPrice)}")

if save:
    # save the predicted prices to a CSV file
    outputDataFrame = pd.DataFrame({
        "Date": dates[:len(finalPredictedPrice)],
        "Predicted " + metric + " Price": finalPredictedPrice.flatten()
        })

    outputCSV = "predicted"+metric+".csv"
    outputDataFrame.to_csv(outputCSV, index=False)
    print(f"Predicted prices saved to {outputCSV}")


    # save the trained model
    from keras.models import load_model

    fileName = "Trained "+metric+" LSTM Model "+str(startDate.day)+"." +str(startDate.month) +"."+str(startDate.year)[2:]+" to "+str(endDate.day)+"." +str(endDate.month) +"."+str(endDate.year)[2:]+".keras"     #cant use "/" in this for some reason
    lstmModel.save(fileName)
    print(f"Model saved as {fileName}")