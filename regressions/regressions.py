"""
    Regressions
"""
import numpy as np
import pandas as pd
from matplotlib import pyplot
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

X = np.array(
    [
        3.3,
        4.4,
        5.5,
        6.71,
        6.93,
        4.168,
        9.779,
        6.182,
        7.59,
        2.167,
        7.042,
        10.791,
        5.313,
        7.997,
        5.654,
        9.27,
        3.1,
    ]
)


Y = np.array(
    [
        1.7,
        2.76,
        2.09,
        3.19,
        1.694,
        1.573,
        3.366,
        2.596,
        2.53,
        1.221,
        2.827,
        3.465,
        1.65,
        2.904,
        2.42,
        2.94,
        1.3,
    ]
)

# Generate the line of best fit with predicted values
predicted_y_values = list(map(lambda x: 0.7 * x + 0.3, X))
given_num_error = sum([(i - j) ** 2 for i, j in zip(Y, predicted_y_values)])

# Print the given number error
print(given_num_error)

print(np.polyfit(X, Y, 1))

# Generate the line of best fit using the polyfit values
predicted_y_values = list(map(lambda x: 0.25 * x + 0.79, X))
polyfit_num_error = sum([(i - j) ** 2 for i, j in zip(Y, predicted_y_values)])

# print polyfit_num_error
print(polyfit_num_error)
print(given_num_error - polyfit_num_error)

# plot the predicted y values
pyplot.scatter(X, Y)
pyplot.plot(X, predicted_y_values)
pyplot.show()

# Use scikit learn to get the line of best fit
lr_reg = LinearRegression()
lr_reg.fit(X.reshape(-1, 1), Y.reshape(-1, 1))

# print out the coefficient and the y intercept
print(lr_reg.coef_)
print(lr_reg.intercept_)

print("Now in the sales section!")
print("Get CSV and create both the x and y columns")
dataframe = pd.read_csv("./Advertising.csv")
feature_cols = ["TV", "radio", "newspaper"]
X = dataframe[feature_cols][:150]
Y = dataframe.sales[:150]

print("Generating a linear regression on the top 150 companies (first one)")
sales_regression = LinearRegression()
sales_regression.fit(X, Y)

print("Printing the coefficient and y intercept of our first 150 sales")
print(sales_regression.coef_)
print(sales_regression.intercept_)

print("Predicting the sales of the bottom 50 companies (last ones)")
print(sales_regression.predict(dataframe[feature_cols][:-50]))

r2_score
