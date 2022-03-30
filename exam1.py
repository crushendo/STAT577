import random
import numpy
import pandas as pd
import statistics
import numpy as np
import matplotlib.pyplot as plt
import math
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.preprocessing import scale
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from numpy import mean
from numpy import absolute
from numpy import sqrt

# Import CSV file as dataframe
lifedf = pd.read_csv("Country_GDP_lifeExp.csv")
'''
#############
#      1a   #
#############
# Scatterplot of predictor vs response
lifedf.plot.scatter(x = 'lifeExp', y = 'gdpPercap', s = 100)
plt.show(block=True)

# Scatterplot of predictor vs response
lifedf.plot.scatter(x = 'lifeExp', y = 'gdpPercap', s = 100)
plt.yscale('log')
plt.show(block=True)

#############
#      1b   #
#############

yr_1952 = lifedf.query('year == 1952', inplace = False)
gdp_1952 = list(yr_1952.loc[:,"gdpPercap"])
print(max(gdp_1952[:]))
print(min(gdp_1952[:]))
max = lifedf.query('gdpPercap == 108382.3529 and year == 1952', inplace = False)
print(max)
min = lifedf.query('gdpPercap == 298.8462121 and year == 1952', inplace = False)
print(min)

# Kuwait has the highest GDP in 1952, while Lesotho had the lowest

#############
#      1c   #
#############

print(lifedf)
cont = 'Europe'
europe = lifedf.query('continent == "Europe"', inplace = False)
europe.plot.scatter(x = 'year', y = 'gdpPercap', s = 100)
plt.show()

# Overall, there appears to be a slight positive increase in GDP over time in European countries

#############
#      1d   #
#############

print(lifedf['continent'].unique)

cont = lifedf.query('continent == "Asia"', inplace=False)
cont.plot.scatter(x='year', y='gdpPercap', s=100)
plt.show()

# The GDP remains fairly steady in asia over time, with some outlier high GDPs falling

cont = lifedf.query('continent == "Africa"', inplace=False)
cont.plot.scatter(x='year', y='gdpPercap', s=100)
plt.show()

#

cont = lifedf.query('continent == "Americas"', inplace=False)
cont.plot.scatter(x='year', y='gdpPercap', s=100)
plt.show()

#

cont = lifedf.query('continent == "Oceania"', inplace=False)
cont.plot.scatter(x='year', y='gdpPercap', s=100)
plt.show()

#############
#      1e   #
#############

years = list(lifedf.loc[:,"year"])
years = (numpy.unique(years))
print(years)

reporting = lifedf.query('year == 1952', inplace=False)
countries = list(reporting.loc[:, "country"])
print(len(numpy.unique(countries)))

reporting = lifedf.query('year == 1957', inplace=False)
countries = list(reporting.loc[:, "country"])
print(len(numpy.unique(countries)))

reporting = lifedf.query('year == 1962', inplace=False)
countries = list(reporting.loc[:, "country"])
print(len(numpy.unique(countries)))
reporting = lifedf.query('year == 1967', inplace=False)
countries = list(reporting.loc[:, "country"])
print(len(numpy.unique(countries)))
reporting = lifedf.query('year == 1972', inplace=False)
countries = list(reporting.loc[:, "country"])
print(len(numpy.unique(countries)))
reporting = lifedf.query('year == 1977', inplace=False)
countries = list(reporting.loc[:, "country"])
print(len(numpy.unique(countries)))
reporting = lifedf.query('year == 1982', inplace=False)
countries = list(reporting.loc[:, "country"])
print(len(numpy.unique(countries)))
reporting = lifedf.query('year == 1987', inplace=False)
countries = list(reporting.loc[:, "country"])
print(len(numpy.unique(countries)))
reporting = lifedf.query('year == 1992', inplace=False)
countries = list(reporting.loc[:, "country"])
print(len(numpy.unique(countries)))
reporting = lifedf.query('year == 1997', inplace=False)
countries = list(reporting.loc[:, "country"])
print(len(numpy.unique(countries)))
reporting = lifedf.query('year == 2002', inplace=False)
countries = list(reporting.loc[:, "country"])
print(len(numpy.unique(countries)))
reporting = lifedf.query('year == 2007', inplace=False)
countries = list(reporting.loc[:, "country"])
print(len(numpy.unique(countries)))

# Over time, the number of countries reporting data has not changed
'''
#############
#      1f   #
#############

# Scatterplot of predictor vs response
lifedf.plot.scatter(x = 'pop', y = 'lifeExp', s = 100)
plt.show(block=True)



#############
#      2a   #
#############

#

#############
#      2b   #
#############

