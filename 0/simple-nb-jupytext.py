# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.3'
#       jupytext_version: 1.0.5
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# + {"pycharm": {}, "cell_type": "markdown"}
# # Intro  
# Welcome to your first programming exercise of this course. We will investigate a diabetes-dataset and see if certain variables are associated with each other. In this tutorial, we want to test for true associations with linear regression.  
#
# Depending on the causal connections between two variables, $x$ and $y$, their true relationship may be linear or nonlinear. However, regardless of the true pattern of association, a linear model can always serve as a first approximation. In this case, the analysis is particularly simple,  
# $$y = \alpha + \beta x + e$$ 
# where $\alpha$ is the y-intercept, $\beta$ is the slope of the line (also known as the regression
# coefficient), and e is the residual error. Letting  
# $$\hat y = \alpha + \beta x$$
# be the value of $y$ predicted by the model, then the residual error is the deviation between the observed and predicted y value, i.e., $e =  y - \hat y$. When information on $x$ is used to predict $y$, $x$ is referred to as the predictor or independent variable and $y$ as the response or dependent variable.
#
# The objective of linear regression analysis is to estimate the model parameters, $\alpha$ and $\beta$, that give the “best fit” for the joint distribution of $x$ and $y$. The true parameters $\alpha$ and $\beta$ are only obtainable if the entire population is sampled. With an incomplete sample, $\alpha$ and $\beta$ are approximated by sample estimators, denoted as $a$ and $b$. In real-world applications, there is often only a weak linear relationship between two variables which makes it difficult to say anything more precise. In order to infer the parameters of our model, an objective definition of “best fit” is required. The mathematical method of least-squares linear regression provides one such best-fit solution.
#
# For this exercise you will need the pandas, matplotlib and numpy python packages.
#
# We provide template code, which you will have to complete. The parts you will have to change are marked with `#your_code`
#
# The ideas conveyed in this exercise, especially the connection between the covariance of two variables $Cov(x,y)$ and the regression coefficient $b$ are explained in more detail [here](http://nitro.biosci.arizona.edu/courses/EEB581-2006/handouts/bivdistrib.pdf).

# + {"pycharm": {}, "cell_type": "markdown"}
# ## The Diabetes Dataset
#
# In this exercise we will be working with the [Pima Indians Diabetes Database](https://www.kaggle.com/uciml/pima-indians-diabetes-databasec).
#
# The dataset consists of 8 medical predictor variables and one target variable, which were gathered in order to predict diabetes outcome in a female population of the native American Pima-tribe (USA). The dataset was part of a kaggle competition.
#
# The data is stored in a CSV-file. We use the pandas package to load the data. An introduction in to pandas can be found [here](https://pandas.pydata.org/pandas-docs/stable/getting_started/10min.html).
#

# + {"pycharm": {"is_executing": false}}
# for data analysis and manipulation
import pandas as pd

# for manipulation of graph and figures
from matplotlib import pyplot as plt

# access to square root function from mathematical library
from math import sqrt

# + {"pycharm": {"is_executing": false}}
# this loads the data into a pandas DataFrame, './diabetes.csv' specifies the directory for the datafile
df = pd.read_csv('./diabetes.csv')

# + {"pycharm": {}}
# this command shows the first rows of the table
df.head()

# + {"pycharm": {}}
# in a jupyter-notebook, you can display the documentation for any function/method/object by propending a question mark:
# ?plt.scatter

# + {"pycharm": {}}
# let's explore the dataset a little bit
# for this exercise we are interested in the relationship between insulin and BMI.
# For this we can produce a scatter-plot of BMI vs Insulin.
# Try to explore other variables as well, using the plt.scatter function

plt.scatter(df.BMI, df.Insulin)
plt.xlabel('BMI')
plt.ylabel('Insulin')
plt.suptitle('BMI vs Insulin')
plt.show()

# + {"pycharm": {}}
# In the plot above, we can see that there are many datapoints which do not have a measurement for Insulin (Insulin == 0).
# We decide that we want to exclude these values. 

# There are also missing values for BMI (BMI == 0), which we exclude as well.

df = df[df.Insulin>0]
df = df[df.BMI>0]
plt.scatter(df.BMI, df.Insulin)
plt.xlabel('BMI')
plt.ylabel('Insulin')
plt.suptitle('BMI vs Insulin')
plt.show()


# + {"pycharm": {}, "cell_type": "markdown"}
# # Step 1: Implementation using lists
# **Task 1.1:** Write functions to caluclate the mean and variance. Call those functions to calculate the mean and variance for BMI and Insulin.
#
# $$Mean, \mu = \frac{\sum x_i}{n} $$
#
# $$Variance, \sigma ^{2} = \frac{\sum (x - \mu)^{2}}{n}$$

# + {"pycharm": {}}
# Calculate the mean value of a python list of numbers
# def is used to define our own functions

def mean(values):
    m = sum(values) / len(values)
    return m
    
# Calculate the variance of a list of numbers
def variance(values, mean):
    v = sum([(x -  mean) ** 2 for x in values])/ len(values)
    return v


# + {"pycharm": {"is_executing": false}}
# we store our variables in lists
x = list(df.BMI)
y = list(df.Insulin)

# + {"pycharm": {"is_executing": false}}
# calculate mean and variance

mean_x, mean_y = mean(x),mean(y)# your_code
var_x, var_y = variance(x, mean_x), variance(y, mean_y) # your_code

print('x stats: mean=%.3f variance=%.3f' % (mean_x, var_x))
print('y stats: mean=%.3f variance=%.3f' % (mean_y, var_y))


# + {"pycharm": {}, "cell_type": "markdown"}
# **Expected output**:  
# x stats: mean = 33.073 variance = 49.210  
# y stats: mean = 155.718 variance = 14096.406

# + {"pycharm": {}, "cell_type": "markdown"}
# Now we want to investigate if the BMI and Insulin are somehow associated. We do this by calculating the covariance:
#
# The covariance is a measure of association between x and y. It is positive if y increases with increasing x, negative if y decreases as x increases, and zero if there is no linear tendency for y to change with x. If x and y are independent, then σ(x,y) = 0, 
#
# $$Cov (x,y) = \frac{\sum(x_{i}-\bar{x})(y_{i}-\bar{y})}{n} $$
#
# **Question 1.1:** Are two variables x, y always independent, if their covariance is 0? Write your anser in the field below by double-clicking.

# + {"pycharm": {}, "active": ""}
# Following the alternate form of the definition (E[XY]-E[X]E[Y]) we know that if Cov(X,Y)=0 then E[XY]=E[X]E[Y]. Since this is only true if the two random variables are independet, the statement above is true.

# + {"pycharm": {}}
# Calculate covariance between x and y
def covariance(x, mean_x, y, mean_y):
    covar = 0.0
    for i in range(len(x)):
        covar += (x[i] - mean_x)*(y[i] - mean_y) # your_code
    return covar / float(len(x))

covar = covariance(x, mean_x, y, mean_y)
print('Covariance: %.3f' % (covar))


# + {"pycharm": {}, "cell_type": "markdown"}
# **Expected output**:  
# Covariance: 189.938

# + {"pycharm": {}, "cell_type": "markdown"}
# Now we want to find coefficients for a line, that predicts our observations best.  
#
# To find the 'best fit' of coefficients for this predictor, we calculate the least-squares linear regression. The least-squares solution yields the values of $a$ and $b$ that minimize the mean squared residual. The formula is the following:
# $$a = \bar y − b * \bar x $$
# $$b = Cov(x, y) / Var(x)$$
# Thus, the least-squares estimators for the intercept and slope of a linear regression are simple functions of the observed means, variances, and covariances.   
#  
# **Task 1.2:** Write a function that calculates the model parameters. Call this function to calculate the coefficients $a$ and $b$ for our linear model to predict the Insulin-level from a BMI observation.

# + {"pycharm": {"is_executing": false}}
# Calculate coefficients a and b
def coefficients(x,y):
    b = covar / var_x # your_code
    a = mean_y - b * mean_x # your_code
    return a, b

a, b = coefficients(x,y)
print('Coefficients: a=%.3f, b=%.3f' % (a, b))


# + {"pycharm": {}, "cell_type": "markdown"}
# **Expected output:**  
# Coefficients: a=28.067, b=3.860

# + {"pycharm": {}, "cell_type": "markdown"}
# Now that we predicited the coefficients for our 'best fit' linear regression model, we need to test it by predicting Insulin-levels from observing BMIs. In other words, if we have a measurement of the BMI, how well can we predict the Insulin level with our model?
#
# To do this, we take our linear regression function, insert our estimated coefficients, and calculate y for each observation x. 
#
# $$\hat y = a + b * x$$
#
#
# **Task 1.3:** Write a function that predicts y from x. Call this function with x and y.
#

# + {"pycharm": {"is_executing": false}}
# Make predictions
def simple_linear_regression(x, y):
    a, b = coefficients(x, y)
    predictions = [a + b * element for element in x]
    return predictions

predictions =simple_linear_regression(x, y) # your_code


# + {"pycharm": {}, "cell_type": "markdown"}
# To evaluate our predictions, we look at the difference between the Insulin level, that our model predicts and the true Insulin level and calculate the root mean squared error (rmse).
#
# $$ RMSE = \sqrt{\frac{\sum (\hat{y_{i}}-y_{i})^2}{n}} $$
#
# **Task 1.4:** Write a function that calculates the root mean squared  error, between the true and predicted value. 

# + {"pycharm": {"is_executing": false}}
 
# Calculate root mean squared error
def rmse_metric(actual, predicted):
    sum_error = 0.0
    for i in range(len(actual)):
        sum_error += (predicted[i] - actual[i]) ** 2
    mean_error = sum_error / len(actual) # your_code
    
    return sqrt(mean_error)


# Evaluate regression algorithm on training dataset
def evaluate_algorithm(x, y, algorithm):
    predicted = algorithm(x, y)# your_code
    rmse = rmse_metric(y, predicted)# your_code
    return rmse

rmse = evaluate_algorithm(x, y, simple_linear_regression)
print('RMSE: %.3f' % (rmse))

# + {"pycharm": {}, "cell_type": "markdown"}
# **Expected output: **   
# RMSE: 115.600

# + {"pycharm": {}, "cell_type": "markdown"}
# Let's plot our results with the pyplot package from matplotlib:  

# + {"pycharm": {"is_executing": false}}
plt.scatter(x, y, color='black')
plt.xlabel('BMI')
plt.ylabel('Insulin')
plt.suptitle('RMSE')
plt.plot(x, predictions, color='blue', linewidth=3)


# + {"pycharm": {}, "cell_type": "markdown"}
# # Discussion:  
# 1) What can you say about the association between BMI and Insulin?  
# 2) Is BMI a good predictor for Insulin?  
# 3) What would be your next steps to improve the prediction for Insulin?  
#
# **Task 1.5:** Write your answers in the cell below

# + {"pycharm": {}, "active": ""}
# 1) The higher the BMI the higher the insulin seems to be. But we saw in the variance a very high value. Thus we know that this correlation is not very strong.
#
# 2) Since the correlation between BMI and Insulin is very weak, the BMI is not a good predictor for Insulin.
#
# 3) You should add additional variables since the prediction in the BMI range of 20-45 is pretty bad. A more complex function than linear would not do anything, since the points are just clustered.

# + {"pycharm": {}, "cell_type": "markdown"}
# # Step 2: Implementation using numpy
#
# Working with lists required us to write a lot of code. Python can save you from all this work, if you are familiar with the numpy package.
# If you are not familiar with numpy yet. We recommend to browse through the documentation page at https://www.numpy.org and also work through the provided tutorial there.  

# + {"pycharm": {}}
# First we need to import the numpy package
import numpy as np

# The 'as np' notion is optional, but it is helpful because we can use the np abbreviation 
# when we call functions from the numpy module. 

# + {"pycharm": {}, "cell_type": "markdown"}
# **Task 2.1:**  
# 1) Define x and y. This time, we do not have to convert them into lists. They can keep it as arrays.  
# 2) Calculate the mean and variance for x and y, using numpy functions. 

# + {"pycharm": {}}
# calculate mean and variance
x = df.BMI
y = df.Insulin

mean_x = np.mean(x)
mean_y = np.mean(y)
var_x = np.var(x)
var_y = np.var(y)

print('x stats: mean=%.3f variance=%.3f' % (mean_x, var_x))
print('y stats: mean=%.3f variance=%.3f' % (mean_y, var_y))


# + {"pycharm": {}, "cell_type": "markdown"}
# **Expected output:  **  
# x stats: mean=33.073 variance=49.210  
# y stats: mean=155.718 variance=14096.406

# + {"pycharm": {}, "cell_type": "markdown"}
# **Task 2.2:** Now rewrite the covariance() function and call the numpy-covariance function within. The numpy-covariance function returns a covariance matrix (2x2) with 4 values. 
#
# **Question 2.1:** Which values does the covariance matrix represent? Which ones are the actual covariance? What are the other values? Write your anser in the cell below.

# + {"pycharm": {}, "active": ""}
# The first column and the first row represent x, the second column and the second row represent y. This means that only at the indices [0][1] and [1][0] we can get the covariance value between x AND y. At [0][0] is the variance of x and at [1][1] is the variance of y.

# + {"pycharm": {}, "cell_type": "markdown"}
# Now, calculate the covariance, using the numpy function cov(). The function returns a covariance matrix, and you need to index the covariance value within the matrix. 
# So, for the covariance() function, index one covariance value and return it. 

# + {"pycharm": {}}
# calculate covariance
def covariance(x,y):
    covar = np.cov(x,y)[0][1] # your_code
    return covar

covariance(x,y)


# + {"pycharm": {}, "cell_type": "markdown"}
# ** Expected output: **  
# 190.42283065898116

# + {"pycharm": {}, "cell_type": "markdown"}
# **Task 2.3:** Now, write a coefficients() function to calculate the coefficients b0 and b1 by using numpy functions. 
# Return a list of b0 and b1. Call the function.

# + {"pycharm": {}}
# Calculate coefficients
def coefficients(x,y):
    x_mean, y_mean = np.mean(x), np.mean(y) # your_code 
    b = covariance(x,y) / np.var(x) # your_code
    a = np.mean(y) - b * np.mean(x) # your_code
    return [a, b]

# calculate coefficients
a, b = coefficients(x,y)
print('Coefficients: a=%.3f, b=%.3f' % (a, b))

# + {"pycharm": {}, "cell_type": "markdown"}
# ** Expected Output: **  
# Coefficients: a=27.741, b=3.870

# + {"pycharm": {}, "cell_type": "markdown"}
# Congratulations, you made it through the first tutorial of this course!  
#
# # Submitting your assignment
#
# Please rename your notebook and send it to machinelearning.dhc@gmail.com.  
# If you have a google account, you can also share your jupyter-file on Google Drive with this eMail address.
#
# Please rename the file to 1_LinRegTut_GROUP_lastname1_lastname2_lastname3.ipynb, and replace GROUP and "lastnameX" with your respective group and last names (+first name initial).
#
# e.g.:
# 1_LinRegTut_LippertG_MontiR_FehrJ_DasP.ipynb
#
# As this is also the first time for us preparing this tutorial, you are welcome to give us feedback to help us improve this tutorial.  
#
# Thank you!  
#
# Jana & Remo

# + {"pycharm": {}}

