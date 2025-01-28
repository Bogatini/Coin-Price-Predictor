import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import style
import numpy as np
from keras.models import load_model
from requests import Session
from datetime import datetime
import json


LSTMModel = load_model("Trained Close LSTM Model 1.1.99 to 27.11.24.keras")

style.use("fivethirtyeight")

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

# Initialize lists for time and price data
time_data = []
real_price_data = []

fig = plt.figure()
ax1 = fig.add_subplot(1,1,1)
ax1.set_ylim(0, 70000)

priceList = []

# Update the plot with new data
def animate(i):
    response = session.get(urlString, params = parameters)
    real_price = json.loads(response.text)["data"]["1"]["quote"]["USD"]["price"]
    real_price_data.append(real_price)
    real_time = json.loads(response.text)["data"]["1"]["quote"]["USD"]["last_updated"]
    time_data.append(datetime.fromisoformat(real_time.replace("Z", "+00:00"))) # make it look nice


    inputVector = np.array(real_price_data)

    inputVector = inputVector.reshape(len(inputVector), 1)

    mean = np.mean(inputVector, axis = 0)
    std = np.std(inputVector, axis = 0)
    standardizedSet = (inputVector - mean) / std                   # norm

    predictedPrice = LSTMModel.predict(standardizedSet, verbose = 0)

    finalPredictedPrice = (predictedPrice * std) + mean

    ax1.clear()
    ax1.plot(time_data, finalPredictedPrice, color = "red", label = "Predicted Price")
    ax1.plot(time_data, real_price_data, color = "blue", label = "Real Price")

    ax1.legend(loc="upper left", fontsize=10)


# Create the animation
ani = animation.FuncAnimation(fig, animate, interval=60000, cache_frame_data=False)  # Update every 60 seconds

# Show the plot
plt.show()
