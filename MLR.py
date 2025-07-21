from sklearn.datasets import make_regression
import pandas as pd
import numpy as np

import plotly.express as px
import plotly.graph_objects as go

from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score

x,y=make_regression(n_samples=100,n_features=2,n_informative=2,n_targets=1,noise=50)
df=pd.DataFrame({'f1':x[:,0],'f2':x[:,1],'t':y})

print(df.head())
fig=px.scatter_3d(df, x='f1', y='f2', z='t')

# fig.show()
print(df.shape)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=2)

from sklearn.linear_model import LinearRegression  #import Linear regression

lr=LinearRegression()
lr.fit(x_train,y_train) # train the model
y_pred=lr.predict(x_test)
# print(y_test.values)

print("MAE",mean_absolute_error(y_test,y_pred))
print("MSE",mean_squared_error(y_test,y_pred))
print("r2_score",r2_score(y_test,y_pred))

x = np.linspace(-5, 5, 10)
y = np.linspace(-5, 5, 10)
xGrid, yGrid = np.meshgrid(y, x)

# # z_final = lr.predict(final).reshape(10,10)

# z = z_final

# final = np.vstack((xGrid.ravel().reshape(1,100),yGrid.ravel().reshape(1,100))).T
     

# fig = px.scatter_3d(df, x='feature1', y='feature2', z='target')

# fig.add_trace(go.Surface(x = x, y = y, z =z ))

# fig.show()
     
print(lr.coef_)
print(lr.intercept_)
