import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from keras.models import load_model
import datetime as dt
import json
from requests import Session
import time

apiKey = "76e42f70-826e-4ffe-8d25-32a0288575df"
urlString = "https://pro-api.coinmarketcap.com/v1/cryptocurrency/quotes/latest"

parameters = {
    "slug": "bitcoin",
    "convert": "USD"
}

headers = {
    "Accepts": "application/json",
    "X-CMC_PRO_API_KEY": apiKey
}

session = Session()
session.headers.update(headers)


LSTMModel = load_model("Trained Close LSTM Model.keras")

response = session.get(urlString, params=parameters)
price = json.loads(response.text)["data"]["1"]["quote"]["USD"]["price"]
#last_updated = json.loads(response.text)["data"]["1"]["quote"]["USD"]["last_updated"]

print(price)

priceArray = [68750, price]

inputVector = np.array(priceArray)

inputVector = inputVector.reshape(len(inputVector), 1)

mean = np.mean(inputVector, axis = 0)
std = np.std(inputVector, axis = 0)



standardizedSet = (inputVector - mean) / std                   # norm

print(standardizedSet.shape)
standardizedSet = standardizedSet.reshape((len(standardizedSet), 2, 1))

predictedPrice = LSTMModel.predict(standardizedSet, verbose = 0)


originalPredictedPrice = (predictedPrice * std) + mean      # inverse

print(priceArray)
print(originalPredictedPrice)

plt.plot(priceArray, color = "red", label = "Predicted Price")
plt.plot(originalPredictedPrice, color = "blue", label = "Real Price")
plt.show()