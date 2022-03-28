import random
import numpy
import pandas as pd
import statistics
import numpy as np
import matplotlib.pyplot as plt
import math
from datetime import datetime

# Import CSV file as dataframe
valuedf = pd.read_csv("VALUE.csv")

# Scatterplot of predictor vs response
valuedf.plot.scatter(x = 'Gender', y = 'CustomerLV', s = 100)
plt.show(block=True)

# Histogram of predictor data
plt.hist(valuedf.loc[:,""], bins = [0, 0.2, 0.4, 0.6, 0.8, 1])
plt.show()

# Box plot
boxplot = valuedf.boxplot(column=['C1', 'C2', 'C3'])
plt.show()

# Get quantile from each column
print(valuedf[''].quantile([.2,.4,.6,.8]))

# Perform stats on predictor
box = list(valuedf.loc[:,"BoxOffice"])
print(statistics.median([x for x in box if x < 177 and x > 128]))
print(max(box[:3049]))
comedy = sum(1 for x in box if x == "Comedy")

# Query data from df
movie2 = moviedf.query('Budget >= 10 and Budget <= 20', inplace = False)

# Remove bad data from predictor list
myList = list(filter(("NA").__ne__, box))