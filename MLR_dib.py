import pandas as pd
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score;

X,y=load_diabetes(return_X_y=True)
# print(X)
# print(y)
# print(X.shape)

X_train,X_test,y_train,y_test=train_test_split(X,y,train_size=0.8,random_state=2)
# print(X_train.shape,X_test.shape)
# lr=LinearRegression()

# lr.fit(X_train,y_train)

# y_pred=lr.predict(X_test)
# # print(y_pred)

# print("MAE",mean_absolute_error(y_test,y_pred))
# print("MSE",mean_squared_error(y_test,y_pred))
# print("r2_score",r2_score(y_test,y_pred))

# print(lr.intercept_)
# print(lr.coef_)

#making our own LR

class myLR:
    def __init__(self):
        self.coef_=None
        self.intercept_=None

    def fit(self,X_train,y_train):
        X_train=np.insert(X_train,0,1,axis=1)
        # print(np.insert(X_train,0,1,axis=1).shape). it add one colom in x_train in 0th col
        betas=np.linalg.inv(np.dot(X_train.T,X_train)).dot(X_train.T).dot(y_train)
        self.intercept_=betas[0]
        self.coef_=betas[1:]
        print(betas)
    def predict(self,X_test):
        y_pred=np.dot(X_test,self.coef_)+self.intercept_
        return y_pred
    
lr=myLR()
lr.fit(X_train,y_train)
y_pred=lr.predict(X_test)
print(lr.predict(X_test))

print(np.insert(X_train,0,1,axis=1).shape)
print("r2_score",r2_score(y_test,y_pred))



