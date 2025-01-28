##
# take a .csv file containing a coin's price data with UNIX timestamps
# and return a slice of said file as a new .csv, with prices between two dates
#
# then show this new file as a graph over time
##

import pandas as pd
import datetime
import time
import csv

import matplotlib.pyplot as plt # for the graph can remove without affecting file read/write stuff

input_csv = "btcusd_1-min_data.csv"   # DONT OPEN THIS - OPENING IT IN EXCEL DELETES BOTTOM LINES
output_csv = "BTCOutput 2020-24.csv"  # CHANGE THIS    - if the number of entries is really high, opening this file can cause data loss JUST DONT LOOK IN THEM

                             # y    m  d  h   m   s
startDate = datetime.datetime(2020, 1, 1, 00, 00, 00) # inclusive
endDate =   datetime.datetime(2024, 1, 1, 00, 00, 00) # exclusive

startDate = time.mktime(startDate.timetuple()) # turn them into unix
endDate = time.mktime(endDate.timetuple())

# Read the CSV file
data = pd.read_csv(input_csv)

mydict = {}

totalVolume = 0

with open(output_csv, "w", newline="") as f:
    with open(input_csv) as file:
        reader = csv.DictReader(file, delimiter=",")
        for row in reader:
            #name = row["Timestamp"]
            #print(row)
            if row["Timestamp"]:
                stamp = int(row["Timestamp"][:-2])
            else:
                break
            #print(stamp)

            if stamp >= startDate and stamp < endDate:   # this is just 2023   #1577836800 = 1/1/2020   1672531200 = 1/1/2023  1704067200 = 1/1/2024

                totalVolume += float(row["Volume"])

                mydict.update(row)

                w = csv.DictWriter(f, mydict.keys())
                #w.writeheader()     # this should only happen once!
                w.writerow(mydict)

print(f"Total volume for time period: {round(totalVolume)}")
print(f"CSV slice saved to {output_csv}\nHere is the data as a graph:")

data = pd.read_csv(output_csv)

x = pd.to_datetime(data.iloc[:,0], unit="s")  # timestamp
y = data.iloc[:,5]  # volume

plt.plot(x, y)
plt.xlabel("Time")
plt.ylabel("Price (USD)")
plt.grid()
plt.show()