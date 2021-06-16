### **Import the following python packages**

```Python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.model_selection from train_test_split
import sklearn.linear_model from LinearRegression
%matplotlib inline
```

Now we need to use pandas to import the CSV into our python script

### **Before applying any statistical techniques to the data set, it is best to extract information about the data set.**

Bellow are some useful python methods for extrapolating information from the data set

`df.describe() %Includes information on count, mean, std and more on the data set`

`df.info()%Number of columns, column name, and data types`

One can also plot several graphs to get a better idea about the distribution of a subset of your data, or correlations between independent variables in your data set.

`sns.histplot(df['name_of_column']) %Creates histogram of a specific column in your dataset`

`sns.heatmap(df.corr(),annot = True) %Creates heatmap of correlation between independent variables`

**The heat map is a great way to find which predictors are associated with the response.*

* * *

### **After getting accustomed to the data set your working with it is time to split your data between your training set and your testing set**

Remember a typical linear regression model has the following equation


$Y = f(X) + \epsilon$

where $X$ contain the dependent variables (or input) variables and $Y$ is the output variables.

**We are going to define out input variables using the following python commands.**

First print our the column names of your data frame

`df.columns %Print out column names of data frame`

*Create a variable containing all the input variables*

`X = df[[List_of_Input_Variables]]`

*Create a variable containing the output variable*

`Y = df[Input_Variable]`

### **Utilize a sklearn splitter function to split your training data and testing data**

*This splitting will involve tuple unpacking*

`x_train, x_test, y_train, y_test = train_test_split(input_variables, output_variable, test_size = 0.4, train_size = 0.6, random_state = 42)`

Now we can instantiate the LinearRegression class so that it requires less typing to call different methods

`lm = LinearRegression()`

We can perform a linear fit on the training data using the .fit() method

`lm.fit(x_train, y_train)`

### Utilizing lm we are able to obtain the coefficients for our linear regression model

Looking at the function below $f(x)$ used for obtaining an approximate value of $Y$. As we can see, $f(x)$ depends on the coefficients $\beta_{1}, \beta_{2}....\beta_{n}$ which is what we are trying to calculate.

$$f(x)=\sum_{i=0}^{N} \beta_{i}x_{i}$$
$$Y\approx f(x)$$

The following code snippet will calculate the coefficients

`lm.coef_`

To organize the data for our coefficients we can create a pandas data frame

`cdf = pd.DataFrame(lm.coef_, index = X.columns(), columns = ['Coeff.'])`

* * *

**Predictions**

Using the predict method within the scikitlearn linear regression class we can use our test data to predict housing prices. The following code performs such task.

`predictions = Lm.predict(x_test)`

Now we want to compare our predicted results (predictions) vs. our test results (y_test) using a scatter plot. If the plot is a linear cluster then we have a good indication that our predictions match our test results.

`plt.scatter(y_test,predictions)`

* * *

**Determining the Appropriate Model**

To determine if the linear regression model was the correct fit, we can create a distribution plot of the residuals between the test values and the predicted values.

`sns.displot(data = (y_test - predictions))`

If the distribution plot is a normal distribution, then we have chosen the correct model. However, if the distribution plot is not a normal distribution, the we must reconsider our model.

* * *

**Regression Metrics**

Utilizing regression metrics we can determine how close our predicted values where to our test values.

There are three common metrics for evaluating and reporting the performance of the regression model.

1.  Mean Squared Error
2.  Root Mean Squared Error
3.  Mean Absolute Error

Code shown below

```Python
def regressionMetrics(testData, predictedValues):
    print('Mean Absolute Error: ', metrics.mean_absolute_error(testData, predictedValues))
    print('Mean Squared Error: ', metrics.mean_squared_error(testData, predictedValues))
    print('Mean Root Squared Error: ', np.sqrt(metrics.mean_squared_error(testData, predictedValues)))
    
  regressionMetrics(y_test, predictions)
  
#Output
Mean Absolute Error:  82288.22251914957
Mean Squared Error:  10460958907.209501
Mean Root Squared Error:  102278.82922291153
```

For more information on how to evaluate these metrics read the link below.

https://towardsdatascience.com/what-are-the-best-metrics-to-evaluate-your-regression-model-418ca481755b
