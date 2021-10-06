import csv
from math import log, e
import matplotlib.pyplot as plt

# Only works for y=ax^b, not y=ax^b + c
class Regression:
    def __init__(self, x_values, y_values):
        self.x_values = x_values
        self.y_values = y_values
        self.lnx_values = [log(x) for x in x_values]
        self.lny_values = [log(y) for y in y_values]
        self.n = len(x_values)
        self.m = (self.n*sum([self.lnx_values[i]*self.lny_values[i] for i in range(self.n)]) - sum(self.lnx_values)*sum(self.lny_values)) \
            / (self.n*sum([x**2 for x in self.lnx_values]) - (sum(self.lnx_values))**2)
        self.b = (sum(self.lny_values) - self.m*sum(self.lnx_values))/self.n
    
    def predict_y(self, x):
        return e**(self.m*log(x) + self.b)
    
    def show_scatter(self, ln=False):
        plt.scatter(self.lnx_values if ln else self.x_values, self.lny_values if ln else self.y_values)
        plt.xlabel('ln(Explanatory variable)' if ln else 'Explanatory variable')
        plt.ylabel('ln(Response variable)' if ln else 'Reponse variable')
        plt.title('Transformed relationship' if ln else 'Original relationship')
        plt.show()

    def __str__(self):
        return f'Regression line: ln(y) = {self.b} + {self.m}*ln(x)'

with open('data.csv') as f:
    reader = csv.reader(f)
    next(reader)
    data = [(float(row[0]), float(row[1])) for row in reader]
    regression_model = Regression([pair[0] for pair in data], [pair[1] for pair in data])
    print(regression_model) # prints the regression line equation
    print(regression_model.predict_y(130)) # prints the expected y value at x=130
    regression_model.show_scatter() # shows a scatter plot of the original data set (pass True as an argument to get transformed data set)