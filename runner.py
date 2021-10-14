import power_regression as pr
import csv

with open('data.csv') as f:
    reader = csv.reader(f)
    next(reader)
    data = [(float(row[0]), float(row[1])) for row in reader]
    regression_model = pr.PowerRegression([pair[0] for pair in data], [pair[1] for pair in data])
    print(regression_model) # prints the regression line equation
    print(regression_model.predict_y(130)) # prints the expected y value at x=130
    regression_model.show_scatter(True) # shows a scatter plot of the original data set (pass True as an argument to get transformed data set)