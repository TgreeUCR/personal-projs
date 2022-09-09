import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.model_selection import train_test_split
#from sklearn.datasets import fetch_california_housing


from sklearn.datasets import load_boston
boston = load_boston()
#from sklearn.datasets import fetch_california_housing
#housing = fetch_california_housing()
print(boston)

df_x = pd.DataFrame(boston.data, columns=boston.feature_names)
df_y = pd.DataFrame(boston.target)

df_x

#initialize linear regression model
reg = linear_model.LinearRegression()

#split the data into 67% training and 33% testing data
x_train, x_test, y_train, y_test=train_test_split(df_x,df_y,test_size=.33,random_state=42)

#training the model with our training data
reg.fit(x_train, y_train)

#print the coefecients/weights for each feature/column of our model
print(reg.coef_)

y_pred= reg.predict(x_test)
print(y_pred)

#print the actual values
print(y_test)

#check the model performance / accuracy using mean squared error or mse and sklearn.metrics
from sklearn.metrics import mean_squared_error
print(mean_squared_error(y_test,y_pred))
