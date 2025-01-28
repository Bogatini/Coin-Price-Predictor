import pandas as pd
import matplotlib.pyplot as plt

def plot_csv(file_path):
    data = pd.read_csv(file_path)

    x = data.iloc[:, 0]  # timestamp
    y = data.iloc[:, 1]  # open price

    plt.figure(figsize=(10, 6))
    plt.plot(x, y)
    plt.xlabel('Time')
    plt.ylabel('Price (USD)')
    plt.grid()
    plt.show()

plot_csv('BTCOutput 2023.csv')