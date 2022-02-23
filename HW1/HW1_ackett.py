import random
import numpy
import pandas as pd
import statistics
import numpy as np
import matplotlib.pyplot as plt
import math
from datetime import datetime

random.seed(10)
moviedf = pd.read_csv("MOVIE.csv")
box = list(moviedf.loc[:,"BoxOffice"])

######################
# Question 1         #
######################

# 1a
print(statistics.median(box))

# 1b
print(statistics.median([x for x in box if x <= 10]))

# 1c
print(statistics.median([x for x in box if x < 177 and x > 128]))

# 1d
topbox = sorted(box, reverse=True)
print(statistics.median(topbox[:14]))

# 1e
ebox = box[:253] + box[254:]
print(statistics.median(ebox))

# 1f
fbox = box[1027:1197]
print(statistics.median(fbox))

# 1g
gbox = box[:1027] + box[1197:]
print(statistics.median(gbox))

######################
# Question 2         #
######################

#2a
print(max(box[:3049]))

#2b
genre = list(moviedf.loc[:,"Genre"])
comedy = sum(1 for x in genre if x == "Comedy")
romance = sum(1 for x in genre if x == "Romance")
thriller = sum(1 for x in genre if x == "Thriller")
print(comedy + romance + thriller)

# 2c
print(sum(box))

#2d
star = list(moviedf.loc[:,"Star"])
print(star[-5:])

# 2e
runtime = list(moviedf.loc[:,"RunningTime"])
print(max(runtime))

# 2f
print(sum(1 for x in box if x > 50) / len(box))

#2g
rows = [325, 864, 1080, 1856, 2164]
cols = [0, 4, 11]
print(moviedf.iloc[rows,cols])

#2h ???????????????
tomato = list(moviedf.loc[:,"Tomatometer"])
y = (sum(1 for x in tomato if x == "NA"))
z = (sum(1 for x in tomato if x != "NA"))
x = [y,z]
print(statistics.mean(x))

# 2i
myList = list(filter(("NA").__ne__, tomato))
myList = list(filter((999.0).__ne__, tomato))
print(np.nanmean(myList))

#2j
movie = list(moviedf.loc[:,"Movie"])
top = max(box)
ind = box.index(top)
print(movie[ind])

######################
# Question 3         #
######################

box = list(moviedf.loc[:,"BoxOffice"])
budget = list(moviedf.loc[:,"Budget"])
performance = []
i = 0
for x in box:
    perind = x / budget[i]
    performance.append(perind)
    i += 1
moviedf["Performance"] = performance

# 3a
performance = list(moviedf.loc[:,"Performance"])
print(sum([x for x in box if x <= 10]) / len(performance))

# 3b
plt.hist(performance, bins = [0, 0.2, 0.4, 0.6, 0.8, 1])
plt.show()

# 3c
logper = [math.log10( x ) for x in performance if x > 0]
plt.hist(logper, bins = [0, 0.2, 0.4, 0.6, 0.8, 1])
plt.show()

######################
# Question 4         #
######################

moviedf.query('Budget >= 10 and Budget <= 20', inplace = True)
moviedf.query('Rating == "G" or Rating == "PG" or Rating == "PG13"', inplace = True)
SUB = moviedf[["Movie", "RunningTime", "Genre", "Tomatometer"]]

# 4a
print(SUB.sort_values(by=['Tomatometer'], ascending=False))

######################
# Question 5         #
######################

buydf = pd.read_csv("BUY.csv")

# 5a
name = list(buydf.loc[:,"name"])
print(len(numpy.unique(name)))

# 5b
print(numpy.unique(name)[0])

######################
# Question 6         #
######################

# 6a
time = list(buydf.loc[:,"buyboxtime"])
pagetimes = []
for date in time:
    formatted = datetime.strptime(date, "%Y-%m-%d %H:%M:%S")
    pagetimes.append(formatted)
print(pagetimes)

# 6b
hours = []
for date in pagetimes:
    hours.append(date.hour)
numhours = list(range(1,24+1))
plt.hist(hours, bins = numhours)
plt.show()

days = []
week_days=["Mon","Tues","Wed","Thur","Fri","Sat","Sun"]
numdays = list(range(0,7))
for date in pagetimes:
    days.append(date.weekday())
plt.hist(days, bins = range(0,7+1), rwidth=0.7)
plt.xticks(numdays, week_days, rotation='vertical')
plt.show()

months = []
monthlabels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
nummonths = list(range(1,12+1))
for date in pagetimes:
    months.append(date.month)
plt.hist(months, bins = range(1,13+1), rwidth=0.7)
plt.xticks(nummonths, monthlabels, rotation='vertical')
plt.show()

# 6c
# The only months in which reviews were counted begin with the number 1. January has the fewest reviews, while Oct,
# Nov, and Dec contain many more months. It is very likely that there was a coding error in the data entry program
# which caused all months to begin with the digit 1. Reviews that were submitted between Feb and Sep were likely
# incorrectly stored in Oct to Dec, causing their much higher quantity of reviews than Jan.

# 6d
# Timestaps for reviews are faily evenly spaced out, with reviews being recorded roughly every 8 hours. This creates
# three peaks in the histogram of reviews by hour of day. However, the time intervals are only roughly every 8 hours,
# ranging often from 7 to 9 hour increments. Therefore, each of the three peaks is its own gaussian curve. It is
# possible that the daemon for collecting incoming reviews was programmed to have three time windows around 5am, 1pm,
# and 9pm. To conserve computing power, the daemon would not be constantly running, but would instead 'wake up' every
# so often and check whether the current time was within one of the time windows, and if so, it would log any new
# reviews. This could have produced the data seen in this dataset.

######################
# Question 7         #
######################

price = list(buydf.loc[:,"price"])
# 7a
i = 0
for thing in price:
    if thing == "NA":
        i += 1
print(i)

# 7b
price_fixed = price
for thing in price_fixed:
    if thing > 100:
        thing = "NA"
median = numpy.median(price_fixed)
print(median)
for thing in price_fixed:
    if thing == "NA":
        thing = median

# 7c
low = list(buydf.loc[:,"low"])
high = list(buydf.loc[:,"high"])
stock = list(buydf.loc[:,"stock"])
sellers = list(buydf.loc[:,"sellers"])

violations = 0
i = 0
for item in low:
    if item > high[i]:
        violations += 1
    i += 1
print(violations)
i = 0
for item in price:
    if item == "NA":
        i += 1
        continue
    if item > high[i] or item < low[i]:
        violations += 1
    i += 1
print(violations)
i = 0
for item in stock:
    if item == "in" or item == "low":
        if sellers[i] == 0:
            violations += 1
    i += 1
print(violations)

