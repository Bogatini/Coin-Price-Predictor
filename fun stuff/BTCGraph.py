##
# DEPRICATED - JUST USE BTC Price Getter.py
#
# takes a .json file containing coin prices and corresponding timestamps
#
# should probs change this to reading a .csv to match up with the other files
# cuz idk how to really use .json files
##

import json
import matplotlib.pyplot as plt
import pandas as pd

# Read data from output.json
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
