# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm:
1. Import pandas, numpy and sklearn
   
2. Calculate the values for the training data set

3. Calculate the values for the test data set

4. Plot the graph for both the data sets and calculate for MAE, MSE and RMSE

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: yogesh rao
RegisterNumber:  212222110055
*/
```

```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error

df=pd.read_csv("C:/classes/ML/ex 2/student_scores.csv")
df.head()
df.tail()
X=df.iloc[:,:-1].values
print("X = ",X)
Y=df.iloc[:,-1].values
print("Y = ",Y)

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=1/3,random_state=0)

#splitting training and test data
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,Y_train)
Y_pred=regressor.predict(X_test)
print("Y_pred = ", Y_pred)
print("Y_test = " , Y_test)

#graph plot for training data
plt.scatter(X_train,Y_train,color="orange")
plt.plot(X_train,regressor.predict(X_train),color="red")
plt.title("Hours vs scores(Training Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

#graph plot for test data
plt.scatter(X_test,Y_test,color="orange")
plt.plot(X_test,regressor.predict(X_test),color="red")
plt.title("Hours vs scores(Test Data Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()


mse=mean_squared_error(Y_test,Y_pred)
print("MSE = ",mse)
mae=mean_absolute_error(Y_test,Y_pred)
print("MAE = ",mae)
rmse=np.sqrt(mse)
print("RMSE : ",rmse)

```

## Output:

![image](https://github.com/Ashwinkumar-03/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118663725/a8371842-e785-46fb-b183-a664ef54ec80)

![image](https://github.com/Ashwinkumar-03/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118663725/e5fb6c4d-1d89-4966-b0d9-86669a5275d2)

![image](https://github.com/Ashwinkumar-03/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118663725/776447ec-a658-42ac-a63a-5ffce9b188c8)

![image](https://github.com/Ashwinkumar-03/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118663725/14c14e77-6765-437e-9362-02177de58a3f)

## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
