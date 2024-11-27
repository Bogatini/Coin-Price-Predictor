import numpy as np
import pandas as pd
from keras.models import load_model
import os
from matplotlib import pyplot as plt
import datetime
import time
import kagglehub

showProgress = 0    # 0 = don't show training  1 = show
showGraph = True
kerasFileName = None # if you want a specific file put it here, else it will use the first model alphabetically
metric = "Close"

# search the directory for any saved keras models
if not (kerasFileName):
    for file in os.listdir():
        if file.endswith(".keras"):
            LSTMModel = load_model(file)
            print(f"Loaded model {file}")
            break # remove this and it will load the last file alphabetically if you wanted that for some reason
else:
    LSTMModel = load_model(kerasFileName)

startDate = datetime.datetime(2024, 1, 1, 00, 00, 00) # inclusive
#endDate = datetime.datetime(2024, 1, 1, 00, 00, 00) # inclusive
endDate = datetime.datetime.now()  # exclusive

startDateUnix = time.mktime(startDate.timetuple()) # turn them into unix timestamps
endDateUnix = time.mktime(endDate.timetuple())

# download latest version of the dataset
path = kagglehub.dataset_download("mczielinski/bitcoin-historical-data")
print("Path to dataset files:", path)

# for an explanation read repeated code in "Coin Price Predictor.py"
csvData = pd.read_csv(path + "\\btcusd_1-min_data.csv", header = 0)

csvData["Datetime"] = pd.to_datetime(csvData["Timestamp"], unit = "s")
timeSlice = csvData[(csvData["Timestamp"] >= startDateUnix) & (csvData["Timestamp"] < endDateUnix)]
timeSlice = timeSlice.copy()
timeSlice["Date"] = timeSlice["Datetime"].dt.date

dataGroup = timeSlice.groupby("Date")
closePriceGroup = dataGroup[metric]
closePriceDataFrame = closePriceGroup.median()

dataSet = closePriceDataFrame.values
dataSet = dataSet.reshape(len(dataSet), 1)

# standardise
mean = np.mean(dataSet, axis=0)
std = np.std(dataSet, axis=0)
standardisedData = (dataSet - mean) / std

predictedData = LSTMModel.predict(standardisedData, verbose = showProgress)

# reveerse standardisation
finalPredictedPrice = (predictedData * std) + mean

# get actual prices for the dates
actualPrices = closePriceDataFrame.values
dates = closePriceDataFrame.index

from sklearn.metrics import mean_squared_error
print(f"The mean squared error is: {mean_squared_error(actualPrices, finalPredictedPrice)}")

plt.plot(dates[:len(finalPredictedPrice)], finalPredictedPrice, color="red", label="Predicted " + metric + " Price", linewidth=0.75)
plt.plot(dates[:len(actualPrices)], actualPrices, color="blue", label="Actual " + metric + " Price", linewidth=0.75)
plt.xticks(rotation=45)
plt.xlabel("Date")
plt.ylabel("Price (USD)")
plt.title("Predicted vs Actual Price")
plt.legend()
plt.show()