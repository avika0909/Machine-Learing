
from sklearn.datasets import make_regression
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score

x,y=make_regression(n_samples=100,n_features=1,n_informative=1,n_targets=1,noise=20)
plt.scatter(x,y)
# plt.show()
# print(x)
from sklearn.linear_model import LinearRegression
lr=LinearRegression()
lr.fit(x,y)
print(lr.coef_)
print(lr.intercept_)

m=lr.coef_
x=np.mean(cross_val_score(lr,x,y,scoring='r2',cv=10))
print(x)

class GDRegressor:
    def __init__(self,learning_rate,epochs):
        
        self.m=100;  # here we can any random value of m and b and try to correct the vealue
        self.b=-120
        self.lr=learning_rate
        self.epochs=epochs
        
    def fit(self,x,y):
        # b using gradient descent
        for i in range(self.epochs):
            loss_slope_b=-2*np.sum(y-self.m*x.ravel()-self.b)
            loss_slope_m=-2*np.sum((y-self.m*x.ravel()-self.b))*x.ravel()
            self.b=self.b-(self.lr*loss_slope_b)
            self.m=self.m-(self.lr*loss_slope_m)
            
            # print(loss_slope,self.b)
        print(self.m,self.b)
    def predict(self,x):
        return self.m*x+self.b
        
        
gd=GDRegressor(0.042,80) #according to requrirements chnage this
gd.fit(x,y)
