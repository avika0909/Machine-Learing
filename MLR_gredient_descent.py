from sklearn.datasets import make_regression
import numpy as np
X,y=make_regression(n_samples=4,n_features=1,n_informative=1,n_targets=1,noise=80,random_state=13)
import matplotlib.pyplot as plt
plt.scatter(X,y)
# plt.show()
from sklearn.linear_model import LinearRegression
lr=LinearRegression()
lr.fit(X,y)

m=lr.coef_
intercept_=lr.intercept_

# print (coef_,intercept_)

plt.scatter(X,y)
plt.plot(X,lr.predict(X),color="red")
# plt.show()

# Lets apply gradient descent for constant m=78.35063668 and take b=0 and go until b=26.159632


y_pred=((78.35*X)+0).reshape(4)

# plt.scatter(X,y)
# plt.plot(X,lr.predict(X),color="red",label="osl")
# plt.plot(X,y_pred,color='blue',label="b=0")
# plt.legend()
# plt.show()

b=0

l=-2*np.sum(y-m*X.ravel()-b)
print(l)

lr1=.1
step_size=lr1*l

print(step_size)

b=b-step_size

y_pred1=((78.35*X)+b).reshape(4)

plt.scatter(X,y)
plt.plot(X,lr.predict(X),color="red",label="osl")
plt.plot(X,y_pred1,color='green',label='b={}'.format(b))
plt.plot(X,y_pred,color='pink',label="b=0")

plt.legend()
plt.show()

# repeat these step until we reach the same line 33 to 51




