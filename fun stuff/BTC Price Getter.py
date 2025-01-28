##
# using the coinmarketcap API to get minute by minute coin price in USD
#
# contains my private API key so idk dont post this online
##

import matplotlib.pyplot as plt
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



#pprint.pprint
#print(json.loads(response.text)["data"]["1"]["quote"]["USD"]["price"])
#print(json.loads(response.text)["data"]["1"]["quote"]["USD"]["last_updated"])


time_data = []
price_data = []

start_time = time.time()

iterations = 5

for x in range(iterations):
    try:
        #API request
        response = session.get(urlString, params=parameters)
        price = json.loads(response.text)["data"]["1"]["quote"]["USD"]["price"]
        last_updated = json.loads(response.text)["data"]["1"]["quote"]["USD"]["last_updated"]

        print(f"Price: {price}, Last Updated: {last_updated}")

        priceTime = last_updated[11:19]

        if priceTime not in time_data:
            time_data.append(priceTime)
            price_data.append(price)

        print(time_data)
        print(price_data)

        time.sleep(60) # new data comes every 60 seconds

    except Exception as e: # idk API probs failed
        print(f"Error occurred: {e}")
        time.sleep(60)


# write to the file here:
output_data = {
    "time_data": time_data,
    "price_data": price_data
}

with open('output.json', 'w') as outfile:
    json.dump(output_data, outfile)



# open what we just made and make a graph
with open('output.json', 'r') as file:
    data = json.load(file)

time_data = data['time_data']
price_data = data['price_data']

# Convert time_data to a list of datetime objects for better plotting
time_data_dt = [pd.to_datetime(time) for time in time_data]

# Create a plot
plt.figure(figsize=(10, 5))
plt.plot(time_data_dt, price_data, marker='o', linestyle='-')

# Add titles and labels
plt.title('Bitcoin Price Over Time')
plt.xlabel('Time (HH:MM)')
plt.ylabel('Price (USD)')
plt.grid()

# Show the plot
plt.tight_layout()
plt.show()
