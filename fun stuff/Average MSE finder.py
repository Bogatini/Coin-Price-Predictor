import subprocess

aaList = []

for x in range(10):
    result = subprocess.run(['python', 'Coin Price Predictor.py'], capture_output=True, text=True)
    aaList.append(float(result.stdout[27:-1]))

print(aaList)

print(f"Mean: {sum(aaList)/len(aaList)}")

# Activation functions:
# tanh:    71000,  50000, 127000 # this isnt good
# sigmoid: 300000, 90000, 216000 # this is shit
# selu:    60000, 123000, 100000 # this is also shit


# optimiser
# adam: 31000, 30000, 30000

