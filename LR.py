import matplotlib.pyplot as plt #for plotting
import pandas as pd
import numpy as np

df=pd.read_csv('placement.csv')
# print(df.head)

plt.scatter(df['cgpa'],df['package']) #plot it x y axis
plt.xlabel('cgpa')
plt.ylabel('package(in lpa)')
plt.grid(True)
# plt.show()

x=df.iloc[:,0:1]
y=df.iloc[:,-1]

# print(x)
# print(y)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=2) #split

from sklearn.linear_model import LinearRegression  #import Linear regression

lr=LinearRegression() #object of LR
lr.fit(x_train,y_train) # train the model

# print(x_test)
# print(y_test)
# print(x_test.iloc[2])

print(lr.predict(x_test.iloc[2].values.reshape(1,1))) #predict
# lr.predict(pd.DataFrame([[8.6]], columns=['cgpa']))

plt.plot(x_test,lr.predict(x_test),color='red') #for line

#to show the line
# plt.legend() 
# plt.show()

m=lr.intercept_ #for y intercept
m=lr.coef_ #for m slope
print(m)

from sklearn.metrics import mean_absolute_error, mean_squared_error,r2_score

y_pred=lr.predict(x_test)
print(y_test.values)

print("MAE",mean_absolute_error(y_test,y_pred))
print("MSE",mean_squared_error(y_test,y_pred))
print("RMSE",np.sqrt(mean_squared_error(y_test,y_pred)))
print("MSE",r2_score(y_test,y_pred)) #how much better your model perform
r2=r2_score(y_test,y_pred)
print(x_test.shape)
x=1-(1-r2)*(41-1)/(41-1-1)
print(x)