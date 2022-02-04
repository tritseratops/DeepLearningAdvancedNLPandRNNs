import numpy as np
import matplotlib.pyplot as plt

# load the data
X = []
Y = []
for line in open("moore.csv"):
    # mark, model, transistors,year, manufacturer, trans_size\
    #     = line.split(' ')
    data = line.split('\t')
    transistors = "".join(data[1].split(','))
    if "[" in transistors:
        transistors, _ = transistors.split('[')
    if " " in transistors:
        _, transistors = transistors.split(' ')
    if "~" in transistors:
        transistors = transistors[1:]
    year = data[2]
    if "[" in year:
        year, _ = year.split('[')
    X.append(float(year))
    Y.append(float(transistors))

# let sturn X and Y into numpy array
X = np.array(X)
Y = np.array(Y)
Y = np.log(Y)
# # plot to see what it looks like
# plt.scatter(X,Y)
# plt.show()

# apply the equations we learned to calculate a and b

denominator = X.dot(X) - X.mean() * X.sum()
a = (X.dot(Y) - Y.mean()*X.sum()) / denominator
b = (Y.mean()*X.dot(X) - X.mean() * X.dot(Y)) / denominator

# calculated predicted Y
Yhat = a*X + b

# plot it all
plt.scatter(X,Y)
plt.plot(X, Yhat)
plt.show()

# calculate r-squared
d1 = Y - Yhat
d2 = Y - Y.mean()
r2 = 1- d1.dot(d1)/d2.dot(d2)
print("The r-squared is: ", r2)

t = np.log(2)/a
# calculate time to double transistor count
print("Time to double transistor count in years:", t)
print("a:", a)