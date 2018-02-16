from data_extract import *
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import numpy as np
import matplotlib.pyplot as plt

steps = [('scaled', StandardScaler()), ('linear_regression', LinearRegression())]

pipeline = Pipeline(steps)

#Create train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

#Compute 5 fold cross validation score
cv_scores = cross_val_scores(pipeline, X, y, cv=5)
print('5 fold cross validation score: {}'.format(cv_scores))
print('Average cross validation score: {}'.format(np.mean(cv_scores)))

#fit the model to the data
pipeline.fit(X_train, y_train)

#compute predictions over the prediction space
y_pred = pipeline.predict(prediction_space)

#R^2
print('R^2: {}'.format(pipeline.score(X_test, y_test)))
#RMSE
rmse np.sqrt(mean_squared_error(y_test, y_pred)
print('Root mean squared error: {}'.format(rmse))

#plot regression line
plt.plot(prediction_space, y_pred, color='black', linewidth=3)
plt.show()
