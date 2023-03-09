# -*- coding: utf-8 -*-
"""
Created on Thu Dec 29 15:17:30 2023

@author: ramav
"""

#importing required librarires
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf
from statsmodels.graphics.regressionplots import influence_plot
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("D:\\Assignments\\Multi linear\\50_Startups.csv")
df

#EDA

df.isnull().sum()

df.shape

df.info()

df.describe()

df.corr()

df.columns

list(df)


sns.distplot(df["R&D Spend"])

sns.distplot(df["Administration"])

sns.distplot(df["Marketing Spend"])

sns.distplot(df["Profit"])

sns.countplot(df["State"])


corr = df[df.columns].corr()
plt.figure(figsize = (10,10))
sns.heatmap(corr,annot = True)


plt.figure(figsize = (14,12))
sns.set_style(style='darkgrid')
sns.pairplot(df)
plt.show()

#Getting dummy values for state

df = pd.get_dummies(df, columns=['State'])
df.head()


df.rename(columns={'Marketing Spend':'Marketing'},inplace=True)
df.rename(columns={'R&D Spend':'RD'},inplace=True)
df.rename(columns={'State_New York':'NewYork'},inplace=True)

# data transformation
#Standardization
SS = StandardScaler()
std = SS.fit_transform(df)
std = pd.DataFrame(std, columns=df.columns)

std.head()

#===========================================================================================================================================================================================================================================================
#model fitting

model = smf.ols('Profit~RD+Administration+Marketing+State_California+State_Florida+NewYork',data = std).fit()
#model testing
# Finding Coefficient parameters
model.params


print(model.tvalues, '\n', model.pvalues)


(model.rsquared,model.rsquared_adj)

model.summary()


#Standard Errors assume that the covariance matrix of the errors is correctly specified.
#The smallest eigenvalue is 6.69e-31. This might indicate that there are
#strong multicollinearity problems or that the design matrix is singular.
model_ad = smf.ols('Profit~Administration',data = std).fit()

model_ad.summary()

#Standard Errors assume that the covariance matrix of the errors is correctly specified.

model_mkt = smf.ols('Profit~Marketing',data = std).fit()
model_mkt.summary()


model_rd = smf.ols('Profit~RD',data = std).fit()
model_rd.summary()


model_sc = smf.ols('Profit~State_California',data=std).fit()
model_sc.summary()


model_sf = smf.ols('Profit~State_Florida',data=std).fit()
model_sf.summary()


model_ny = smf.ols('Profit~NewYork',data=std).fit()
model_ny.summary()

#===========================================================================================================================================================================================================================================================
#Variance inflation factor
# Model Validation
# Two Techniques: 1. Collinearity Check & 2. Residual Analysis
# 1) Collinearity Problem Check
# Calculate VIF = 1/(1-Rsquare) for all independent variables


rsq_rd = smf.ols('RD~Administration+Marketing+State_California+State_Florida+NewYork',data=std).fit().rsquared  
vif_rd = 1/(1-rsq_rd)

rsq_ad = smf.ols('Administration~RD+Marketing+State_California+State_Florida+NewYork',data=std).fit().rsquared  
vif_ad = 1/(1-rsq_ad)

rsq_mkt = smf.ols('Marketing~RD+Administration+State_California+State_Florida+NewYork',data=std).fit().rsquared  
vif_mkt = 1/(1-rsq_mkt)

rsq_sc = smf.ols('State_California~RD+Administration+Marketing+State_Florida+NewYork',data=std).fit().rsquared  
vif_sc = 1/(1-rsq_sc)

rsq_sf = smf.ols('State_Florida~RD+Administration+Marketing+State_California+State_Florida+NewYork',data=std).fit().rsquared  
vif_sf = 1/(1-rsq_sf)

rsq_ny = smf.ols('NewYork~RD+Administration+Marketing+State_California+State_Florida',data=std).fit().rsquared  
vif_ny = 1/(1-rsq_ny)

data = {'Features':['RD','Administration','Marketing','State_California','State_Florida','NewYork'],'VIF':[vif_rd,vif_ad,vif_mkt,vif_sc,vif_sf,vif_ny]}
Vif_frame = pd.DataFrame(data)
Vif_frame

#=========================================================================================================================================================
#========================================================================================================================================================

model_Rd_M = smf.ols('Profit~RD+Marketing',data = std).fit()
model_Rd_M.summary()



(model_mkt.rsquared,model_mkt.aic)


(model_rd.rsquared,model_rd.aic)


(model_rd.rsquared,model_rd.rsquared_adj)

#===========================================================================================================================================================================================================================================================

import statsmodels.api as sm
qqplot=sm.qqplot(model_rd.resid,line='q') 
plt.title("Normal Q-Q plot of residuals")
plt.show()

list(np.where(model_rd.resid<-0.8))

#===========================================================================================================================================================================================================================================================

def get_standardized_values( vals ):
    return (vals - vals.mean())/vals.std()
plt.scatter(get_standardized_values(model_rd.fittedvalues),get_standardized_values(model_rd.resid))

plt.title('Residual Plot')
plt.xlabel('Standardized Fitted values')
plt.ylabel('Standardized residual values')
plt.show()


fig = plt.figure(figsize=(12,8))
fig = sm.graphics.plot_regress_exog(model_rd, "RD", fig=fig)
plt.show()



fig = plt.figure(figsize=(12,8))
fig = sm.graphics.plot_regress_exog(model_mkt,"Marketing", fig=fig)
plt.show()


#===========================================================================================================================================================================================================================================================

model_influence = model_rd.get_influence()
(c, _) = model_influence.cooks_distance
(np.argmax(c),np.max(c))


df=std.drop(std.index[[49]],axis=0).reset_index()
df.shape


df.head()


final_model =smf.ols('Profit~RD',data = df).fit()
(final_model.rsquared,final_model.rsquared_adj)


newdata=pd.DataFrame({'RD':100000},index=[1])
final_model.predict(newdata)