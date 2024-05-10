# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1. Import the standard Libraries.
 
2.Set variables for assigning dataset values.

3.Import linear regression from sklearn.

4.Assign the points for representing in the graph.

5.Predict the regression for marks by using the representation of the graph.

6.Compare the graphs and hence we obtained the linear regression for the given datas.

   

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: RITHISH P
RegisterNumber:212223230173  
*/
```
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from sklearn.metrics import mean_absolute_error,mean_squared_error

df=pd.read_csv('student_scores.csv')

df.head()

df.tail()

x = df.iloc[:,:-1].values

x

y = df.iloc[:,1].values

y

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)

from sklearn.linear_model import LinearRegression

regressor = LinearRegression()

regressor.fit(x_train,y_train)

y_pred = regressor.predict(x_tes)

y_pred

y_test

#Graph plot for training data

plt.scatter(x_train,y_train,color='black')

plt.plot(x_train,regressor.predict(x_train),color='purple')

plt.title("Hours vs Scores(Training set)")

plt.xlabel("Hours")

plt.ylabel("Scores")

plt.show()

#Graph plot for test data

plt.scatter(x_test,y_test,color='red')

plt.plot(x_train,regressor.predict(x_train),color='blue')

plt.title("Hours vs Scores(Testing set)")

plt.xlabel("Hours")

plt.ylabel("Scores")

plt.show()

mse=mean_absolute_error(y_test,y_pred)

print('MSE = ',mse)

mae=mean_absolute_error(y_test,y_pred)

print('MAE = ',mae)

rmse=np.sqrt(mse)

print("RMSE= ",rmse)

## Output:
               df.head()
![ml2 1](https://github.com/RITHISHlearn/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/145446645/4b472129-c44e-4df9-85cc-4a8508b2c343)

               df.tail()
![ml2 2](https://github.com/RITHISHlearn/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/145446645/b7ed5586-c651-4643-b710-afb43731237a)

               Array of value of X
![ml2 3](https://github.com/RITHISHlearn/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/145446645/73e10e66-975c-4356-b449-debb75b31dfa)

               Array of value of Y
![ml2 4](https://github.com/RITHISHlearn/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/145446645/33e513b4-f889-4724-abbe-136c1a22eac1)

               Values of Y prediction
![ml2 5](https://github.com/RITHISHlearn/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/145446645/a02febc2-4548-477f-80cd-0e4bdadfc422)

               Array values of Y test
![ml2 6](https://github.com/RITHISHlearn/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/145446645/0e069bd0-224f-4629-bd5a-fe3052abc447)

              Training Set Graph
![ml2 7](https://github.com/RITHISHlearn/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/145446645/bdf1412e-ff9d-4f36-9f1b-edf8cde742d9)

              Test Set Graph
![ml2 8](https://github.com/RITHISHlearn/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/145446645/accfd9cd-d3ce-45e7-a0eb-1fcce9f059c6)

              Values of MSE, MAE and RMSE
![ml2 9](https://github.com/RITHISHlearn/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/145446645/d057c6bb-f0f9-4013-ad88-17fd03a066d4)




## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
