from sklearn.datasets import load_diabetes

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
import time
X,y = load_diabetes(return_X_y=True)
# print(X.shape)
# print(y.shape)

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=2)
reg = LinearRegression()
reg.fit(X_train,y_train)

print(reg.coef_)
print(reg.intercept_)

y_pred = reg.predict(X_test)
r2_score(y_test,y_pred)

print(X_train.shape)


class GDRegressor:
    def __init__(self,lr=0.1,epochs=100):
        self.coef_=None
        self.intercept_=None
        self.lr=lr
        self.epochs=epochs
    def fit(self,X_train,y_train):
        self.intercept_=0
        self.coef_=np.ones(X_train.shape[1])
        print(self.intercept_,self.coef_)

        for i in range(self.epochs):
            #update all coef and intercept valeue
            y_hat=np.dot(X_train,self.coef_)+self.intercept_
            intercept_der=-2*np.mean(y_train-y_hat)
            self.intercept_=self.intercept_-(self.lr*intercept_der)

            coef_der = -2 * np.dot((y_train - y_hat),X_train)/X_train.shape[0]
            self.coef_ = self.coef_ - (self.lr * coef_der)

        #    print(y_hat.shape)

        print(self.intercept_,self.coef_)
        




    def predict(self,X_test):
        return np.dot(X_test,self.coef_) + self.intercept_



gdr=GDRegressor()
start=time.time()
gdr.fit(X_train,y_train)
print("time taken",time.time()-start)

y_pred=gdr.predict(X_test)

a=r2_score(y_test,y_pred)
print(a)




