import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

data = datasets.load_diabetes()
print(data.DESCR)

X, y = datasets.load_diabetes(return_X_y=True)
print(X.shape)
print(y.shape)

X = X[:20,[9]]
y = y[:20]

# 75% train 25% test
X_train, X_test, y_train, y_test = train_test_split(X, y)
#X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.9, test_size=0.1)
print(X_train.shape)
print(X_test.shape)

#fitting the linear regression using only training data
regr = LinearRegression(fit_intercept=True)
regr.fit(X_train,y_train)

print('Coefficients', regr.coef_)
print('Intercept', regr.intercept_)

y_pred_train = regr.predict(X_train)
print(y_pred_train)

y_pred_test = regr.predict(X_test)
print(y_pred_test)
print(y_test)

MSE_train = mean_squared_error(y_train,y_pred_train)
print(MSE_train)
MSE_test = mean_squared_error(y_test,y_pred_test)
print(MSE_test)

r2Score_train = r2_score(y_train, y_pred_train)
print(r2Score_train)

plt.scatter(X_train, y_train, color='black',label='Train data points')
plt.scatter(X_test, y_test, color='red', label = 'Test data points')
plt.scatter(X_test, y_pred_test, marker='x', color='red',label = 'Test Predictions')

# ADDITION: Plot regression line over entire X range
x_line = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
y_line = regr.predict(x_line)
plt.plot(x_line, y_line, color='blue', label='Regression Line')

plt.legend()
plt.show()

X_train_mean = np.mean(X_train, axis=0)
X_train_sd = np.std(X_train, axis=0)
y_mean = np.mean(y)
y_std = np.std(y)

X_train_norm = (X_train - X_train_mean)/X_train_sd 
y_norm = (y_train - y_mean)/y_std

regr.fit(X_train_norm,y_norm)

print('Coefficients', regr.coef_)
print('Intercept', regr.intercept_)

X_test_norm = (X_test - X_train_mean)/X_train_sd 
y_test_norm = (y_test - y_mean)/y_std

# Get small subset of the diabetes dataset from scikit-learn
from sklearn.linear_model import Ridge

X, y = datasets.load_diabetes(return_X_y=True)
print(X.shape)
print(y.shape)

X_train, X_test, y_train, y_test = train_test_split(X,y)

regr = LinearRegression(fit_intercept=True)
regr.fit(X_train,y_train)

print('coefficient', regr.coef_)
print('Intercept', regr.intercept_)

y_pred_test = regr.predict(X_test)
print(y_pred_test.shape)

# MSE test
MSE_test = mean_squared_error(y_test, y_pred_test)
print(MSE_test)

# R2 test
r2score_test = r2_score(y_test, y_pred_test)
print(r2score_test)

# (XTX+lambda I )^-1
Ridge_model = Ridge(alpha=100000, fit_intercept=True).fit(X_train, y_train)

print('coefficients Ridge', Ridge_model.coef_)
print('Intercept Ridge', Ridge_model.intercept_)

y_pred_test_Ridge = Ridge_model.predict(X_test)

MSE_Ridge_Test = mean_squared_error(y_test, y_pred_test_Ridge)
print(MSE_Ridge_Test)
