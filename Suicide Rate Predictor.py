import pandas as pd
import numpy as np
#import sklearn as sk
from sklearn import preprocessing
import warnings as w
from sklearn.metrics import *
#from sklearn.linear_model import Ridge, ElasticNet, Lasso
from sklearn.linear_model import *
from sklearn.model_selection import train_test_split
w.filterwarnings('ignore')
from math import sqrt

#reading dataset
data = pd.read_csv('master.csv')
#print(data.columns)

#renaming columns because spaces were involve in column names in dataset which were creating difficulty in handling of data
renameColumns = ['country','year','sex','age','suicides_no','population','suicides/100k pop','countryYear','HDIforYear', 'gdpforYear','gdpperCapita','generation']
data.columns = renameColumns

#because it is redundant column, we can get values of this column from country and year column            
data = data.drop(['countryYear'],1) 

#print(data.isnull().any())

#HDIforYear column contain null so we are replacing the null data values with mean value of this column
mean = data['HDIforYear'].mean()
#print(mean)
data['HDIforYear'] = data['HDIforYear'].replace(np.NAN,mean)
#print(data['HDIforYear'])


#Using dummy encoding method to make new columns for string data
#creates dummy dataset for sex column
dummies_sex = pd.get_dummies(data.sex)
#merge the dummy dataset which contains male and female column
new_dataset= pd.concat([data,dummies_sex],axis='columns')
#deletes the sex column and also one random dummy column to avoid dummy variable trap
new_dataset = new_dataset.drop(['sex','male'],axis='columns')

#create dummy dataset for country column
dummies_country = pd.get_dummies(data.country)
new_dataset= pd.concat([new_dataset,dummies_country],axis='columns')
new_dataset = new_dataset.drop(['country','Albania'],axis='columns')

#create dummy dataset for age
dummies_age = pd.get_dummies(data.age)
new_dataset= pd.concat([new_dataset,dummies_age],axis='columns')
new_dataset = new_dataset.drop(['age','15-24 years'],axis='columns')

#create dummy dataset for generation
dummies_generation = pd.get_dummies(data.generation)
new_dataset= pd.concat([new_dataset,dummies_generation],axis='columns')
new_dataset = new_dataset.drop(['generation','Generation X'],axis='columns')

#using this for loop to remove commas from gdpforYear column and to convert it into integer value
for i in range(len(new_dataset.gdpforYear)):
	new_dataset.gdpforYear[i]=int(new_dataset.gdpforYear[i].replace(',',''))
#print(new_dataset)

#print(new_dataset.corr())

#target column of the dataset
y = np.array(new_dataset['suicides/100k pop'])
#features of the dataset
X = np.array(new_dataset.drop(['suicides/100k pop'],1))


X_train, X_test, y_train, y_test =  train_test_split(X,y , test_size = 0.33, random_state = 42)


'''
ridge = Ridge(alpha=1, fit_intercept=True, normalize=False, copy_X=True, max_iter=None,tol=0.000001 ,solver='auto', random_state=None)
#ridge = Ridge(alpha=10.0, fit_intercept=True, normalize=False, copy_X=True, max_iter=1000, tol=0.001, solver='sag', random_state=None)
ridgeTrained = ridge.fit(X_train, y_train)
#print(ridgeTrained)
y_hat = ridgeTrained.predict(X_test)
#print(y_hat)

print("RIDGE")
print("Root Mean Squared Error: ",sqrt(mean_squared_error(y_hat, y_test)))
print("Max Error: ",(max_error(y_hat, y_test)))
print("Explained var score: ",explained_variance_score(y_hat, y_test))
print("Mean abs error: ",mean_absolute_error(y_hat, y_test))
print("Mean sq error: ",mean_squared_error(y_hat, y_test))
print("Median abs error: ", median_absolute_error(y_hat, y_test))
print("r2_score: ",r2_score(y_hat, y_test))
'''

print("ELASTIC NET")
elastic = ElasticNet(alpha=0.00001, l1_ratio=0.000005, fit_intercept=True, normalize=False, precompute=False, max_iter=1000, copy_X=True, tol=0.000001, warm_start=False, positive=False, random_state=None, selection='cyclic')
elasticTrained = elastic.fit(X_train, y_train)
#print(elasticTrained.coef_)
y_hat = elasticTrained.predict(X_test)
#print(y_hat[0:500])
#print(y_hat)

print("Root Mean Squared Error: ",sqrt(mean_squared_error(y_hat, y_test)))
print("Max Error: ",(max_error(y_hat, y_test)))
print("Explained var score: ",explained_variance_score(y_hat, y_test))
print("Mean abs error: ",mean_absolute_error(y_hat, y_test))
print("Mean sq error: ",mean_squared_error(y_hat, y_test))
print("Median abs error: ", median_absolute_error(y_hat, y_test))
print("r2_score: ",r2_score(y_hat, y_test))

'''
print("LASSO")
lasso = Lasso(alpha=0.0001 ,fit_intercept=True, normalize=False, precompute=False, copy_X=True, max_iter=1000, tol=0.0001, warm_start=False, positive=False, random_state=None, selection='cyclic')
lassoTrained = lasso.fit(X_train, y_train)
y_hat = lassoTrained.predict(X_test)
#print(y_hat)

print("Root Mean Squared Error: ",sqrt(mean_squared_error(y_hat, y_test)))
print("Max Error: ",(max_error(y_hat, y_test)))
print("Explained var score: ",explained_variance_score(y_hat, y_test))
print("Mean abs error: ",mean_absolute_error(y_hat, y_test))
print("Mean sq error: ",mean_squared_error(y_hat, y_test))
print("Median abs error: ", median_absolute_error(y_hat, y_test))
print("r2_score: ",r2_score(y_hat, y_test))

'''


'''

lasso = RidgeCV(alphas=(0.1, 1.0, 10.0), fit_intercept=True, normalize=False, scoring=None, cv=None, gcv_mode=None, store_cv_values=False)
lassoTrained = lasso.fit(X_train, y_train)
y_hat = lassoTrained.predict(X_test)
#print(y_hat)
print("Mean Squared Error of RidgeCV: ",mean_squared_error(y_hat, y_test))
#print("Max Error of RidgeCV: ",(max_error(y_hat, y_test)))


lasso = BayesianRidge(n_iter=300, tol=0.001, alpha_1=1e-06, alpha_2=1e-06, lambda_1=1e-06, lambda_2=1e-06, compute_score=False, fit_intercept=True, normalize=False, copy_X=True, verbose=False)
lassoTrained = lasso.fit(X_train, y_train)
y_hat = lassoTrained.predict(X_test)
#print(y_hat)
print("Mean Squared Error of Bayesian: ",mean_squared_error(y_hat, y_test))
#print("Max Error of Bayesian: ", max_error(y_hat, y_test))

lasso = SGDRegressor(loss='squared_loss', penalty='l2', alpha=1, l1_ratio=0.15, fit_intercept=True, max_iter=1000, tol=0.001, shuffle=True, verbose=0, epsilon=0.1, random_state=None, learning_rate='invscaling', eta0=0.01, power_t=0.25, early_stopping=False, validation_fraction=0.1, n_iter_no_change=5, warm_start=False, average=False)
lassoTrained = lasso.fit(X_train, y_train)
y_hat = lassoTrained.predict(X_test)
#print(y_hat)
print("Mean Squared Error of Lasso: ",mean_squared_error(y_hat, y_test))
#print("Max Error of SGD: ", max_error(y_hat, y_test))

'''
