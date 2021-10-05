import csv
from math import log, e

# Only works for y=ax^b, not y=ax^b + c
class Regression:
    def __init__(self, x_values, y_values):
        self.n = len(x_values)
        self.m = (self.n*sum([x_values[i]*y_values[i] for i in range(self.n)]) - sum(x_values)*sum(y_values)) \
            / (self.n*sum([x**2 for x in x_values]) - (sum(x_values))**2)
        self.b = (sum(y_values) - self.m*sum(x_values))/self.n
    
    def predict_y(self, x):
        return e**(self.m*log(x) + self.b)

    def __str__(self):
        return f'Regression line: ln(y) = {self.b} + {self.m}*ln(x)'

with open('data.csv') as f:
    reader = csv.reader(f)
    next(reader)
    data = [(float(row[0]), float(row[1])) for row in reader]
    regression_model = Regression([log(pair[0]) for pair in data], [log(pair[1]) for pair in data])
    print(regression_model) # prints the regression line equation
    print(regression_model.predict_y(15)) # prints the expected y value at x=15