import random
import numpy
import pandas as pd
import statistics
import numpy as np
import matplotlib.pyplot as plt
import math
from datetime import datetime

random.seed(577)
valuedf = pd.read_csv("VALUE.csv")

######################
# Question 1         #
######################

# 1a
valuedf.plot.scatter(x = 'Gender', y = 'CustomerLV', s = 100)
plt.show(block=True)
valuedf.plot.scatter(x = 'Married', y = 'CustomerLV', s = 100)
plt.show(block=True)
valuedf.plot.scatter(x = 'Income', y = 'CustomerLV', s = 100)
plt.show(block=True)
valuedf.plot.scatter(x = 'FirstPurchase', y = 'CustomerLV', s = 100)
plt.show(block=True)
valuedf.plot.scatter(x = 'LoyaltyCard', y = 'CustomerLV', s = 100)
plt.show(block=True)
valuedf.plot.scatter(x = 'WalletShare', y = 'CustomerLV', s = 100)
plt.show(block=True)
valuedf.plot.scatter(x = 'TotTransactions', y = 'CustomerLV', s = 100)
plt.show(block=True)
valuedf.plot.scatter(x = 'LastTransaction', y = 'CustomerLV', s = 100)
plt.show(block=True)