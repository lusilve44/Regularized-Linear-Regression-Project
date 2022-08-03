
import pandas as pd 
import numpy as np 
import pickle
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.linear_model import Lasso
from sklearn import metrics
from sklearn.model_selection import GridSearchCV

df = pd.read_csv('/workspace/Regularized-Linear-Regression-Project-Tutorial/data/df_data.csv')

names = df[['fips', 'COUNTY_NAME', 'STATE_NAME']]
targets = df[['ICU Beds_x','Internal Medicine Primary Care (2019)']]
df.drop(['CNTY_FIPS','fips','Active Physicians per 100000 Population 2018 (AAMC)','Total Active Patient Care Physicians per 100000 Population 2018 (AAMC)', 'Active Primary Care Physicians per 100000 Population 2018 (AAMC)', 'Active Patient Care Primary Care Physicians per 100000 Population 2018 (AAMC)','Active General Surgeons per 100000 Population 2018 (AAMC)','Active Patient Care General Surgeons per 100000 Population 2018 (AAMC)','Total nurse practitioners (2019)','Total physician assistants (2019)','Total physician assistants (2019)','Total Hospitals (2019)','Internal Medicine Primary Care (2019)','Family Medicine/General Practice Primary Care (2019)','STATE_NAME','COUNTY_NAME','ICU Beds_x','Total Specialist Physicians (2019)'], axis=1, inplace=True)

X = df
y1 = targets['ICU Beds_x']
y2=  targets['Internal Medicine Primary Care (2019)']


# with ICU Beds_x

X_train, X_test, y_train, y_test = train_test_split(X, y1, test_size=0.3, random_state=42)
pipeline1 = make_pipeline(StandardScaler(), Lasso(alpha=5))
pipeline1.fit(X_train, y_train)
coef_list=pipeline1[1].coef_
loc=[i for i, e in enumerate(coef_list) if e != 0]
col_name=df.columns

X_ols = X_train[col_name[loc]]
X_ols_int = sm.add_constant(X_ols) 
ols_model1 = sm.OLS(y_train, X_ols_int)
results1 = ols_model1.fit()


# with Internal Medicine Primary Care (2019)

X_train, X_test, y_train, y_test = train_test_split(X, y2, test_size=0.3, random_state=42)
pipeline2 = make_pipeline(StandardScaler(), Lasso(alpha=5))
pipeline2.fit(X_train, y_train)
coef_list=pipeline2[1].coef_
loc=[i for i, e in enumerate(coef_list) if e != 0]
col_name=df.columns

X_ols = X_train[col_name[loc]]
X_ols_int = sm.add_constant(X_ols) 
ols_model2 = sm.OLS(y_train, X_ols_int)
results2 = ols_model2.fit()


# saving models  

filename='/workspace/Regularized-Linear-Regression-Project-Tutorial/models/lasso_model1.sav'
pickle.dump(pipeline1, open(filename, 'wb'))

filename='/workspace/Regularized-Linear-Regression-Project-Tutorial/models/lasso_model2.sav'
pickle.dump(pipeline2, open(filename, 'wb'))

filename='/workspace/Regularized-Linear-Regression-Project-Tutorial/models/ols_model1.sav'
pickle.dump(ols_model1, open(filename, 'wb'))

filename='/workspace/Regularized-Linear-Regression-Project-Tutorial/modelsgit/ols_model2.sav'
pickle.dump(ols_model2, open(filename, 'wb'))