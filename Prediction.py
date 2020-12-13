import pandas as pd
import os
os.chdir("C:/Users/flora/Desktop/2021/big_data/project")
df=pd.read_excel('data.xlsx',index_col=0,parse_dates=True,sheet_name="Monthly")
print(df.head())
print(df.columns)
import matplotlib.pyplot as plt
def plots(y_test,y_pred_test):
    plt.figure(figsize=(7,4))
    fig,ax1=plt.subplots()
    plt.plot(y_test.values.reshape(-1,1),'b',lw=1.5,label='Real_return')
    plt.grid(1)
    plt.legend(loc=2)
    plt.title('Real vs Predicted')
    plt.ylabel('Real_return')
    plt.axis('tight')
    ax2=ax1.twinx()
    plt.plot(y_pred_test,'r',lw=1.5,label='predicted_return')
    plt.ylabel('predicted_return')
    plt.legend(loc=0)
#process data
import numpy as np
df['tms']=df.lty-df.rf
df['d_p']=np.log(df.D12/df.Index)
df['e_p']=np.log(df.E12/df.Index)
df['dfy']=df.BAA-df.AAA
df['mktrf']=df.mkt-df.rf
var=df.drop(labels=['Index','D12','E12','AAA', 'BAA','lty','rf','ltr', 'corpr','mkt','csp'],axis=1)
print(var.head())
print(var.columns)

#Statistics Summary
print(var.describe().transpose())
#Nullity or missing values by columns
import missingno as msno
msno.matrix(df=var,figsize=(8,4))

#Correlation
import seaborn as sns
plt.figure(figsize=(12,12))
cor=var.corr()
a=sns.heatmap(cor,annot=True,cmap=plt.cm.Blues,square=True)
plt.show()

#model 1:ElasticNet
from sklearn.model_selection import train_test_split
var['lead_exret']=var.mktrf.shift(-1)
# print(var.lead_exret.head())
var=var[~np.isnan(var).any(axis=1)]
X_train,X_test,y_train,y_test=train_test_split(var.iloc[:,:14],var['lead_exret'],test_size=0.3,shuffle=False,random_state=0)
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
X_train_std=scaler.fit_transform(X_train)
X_test_std=scaler.transform(X_test)
y_train_std=scaler.fit_transform(y_train.values.reshape(-1,1))
y_test_std=scaler.transform(y_test.values.reshape(-1,1))

from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error 
from sklearn.metrics import mean_absolute_error
output=[]
for i in [0.001,0.005,0.01,0.1,1]:
  en=ElasticNet(alpha=i)
  en.fit(X_train_std,y_train)
  y_pred_train=en.predict(X_train_std)
  y_pred_test=en.predict(X_test_std)
  temp=[i,en.score(X_train_std,y_train),mean_squared_error(y_train,y_pred_train),mean_absolute_error(y_train,y_pred_train),en.score(X_test_std,y_test),mean_squared_error(y_test,y_pred_test),mean_absolute_error(y_test,y_pred_test)]
  output.append(temp)
output=pd.DataFrame(output,columns=['alpha','Train_R2','Train_MSE','Train_MAE','Test_R2','Test_MSE','Test_MAE'])
output.to_csv('ElasticNet.csv',index=False)
plots(y_test,y_pred_test)

#model 2:PLS
from sklearn.cross_decomposition import PLSRegression
output=[]
for i in range(1,12,2):
    pls=PLSRegression(n_components=i,max_iter=10000)
    pls.fit(X_train_std,y_train)
    y_pred_train=pls.predict(X_train_std)
    y_pred_test=pls.predict(X_test_std)
    temp=[i,pls.score(X_train_std,y_train),mean_squared_error(y_train,y_pred_train),mean_absolute_error(y_train,y_pred_train),pls.score(X_test_std,y_test),mean_squared_error(y_test,y_pred_test),mean_absolute_error(y_test,y_pred_test)]
    output.append(temp)
output=pd.DataFrame(output,columns=['Components','Train_R2','Train_MSE','Train_MAE','Test_R2','Test_MSE','Test_MAE'])
output.to_csv('PLS.csv',index=False)
pls=PLSRegression(n_components=1,max_iter=10000)
pls.fit(X_train_std,y_train)
y_pred_test=pls.predict(X_test_std)
plots(y_test,y_pred_test)

#model 3:Random Forest
from sklearn.ensemble import RandomForestRegressor
output=[]
for i in [10,100,1000,5000,10000]:
    for j in [5,10,15,20,25,30]:
        rfr=RandomForestRegressor(n_estimators=i,max_depth=j)
        rfr.fit(X_train, y_train)
        y_pred_train=rfr.predict(X_train)
        y_pred_test=rfr.predict(X_test)
        temp=[i,j,rfr.score(X_train,y_train),mean_squared_error(y_train,y_pred_train),mean_absolute_error(y_train,y_pred_train),rfr.score(X_test,y_test),mean_squared_error(y_test,y_pred_test),mean_absolute_error(y_test,y_pred_test)]
        output.append(temp)
output=pd.DataFrame(output,columns=['Num_Tree','Depth','Train_R2','Train_MSE','Train_MAE','Test_R2','Test_MSE','Test_MAE'])
output.to_csv('Forest.csv',index=False)
rfr=RandomForestRegressor(n_estimators=100,max_depth=5)
rfr.fit(X_train, y_train)
y_pred_test=rfr.predict(X_test)
plots(y_test,y_pred_test)
#feature importance
importance=pd.DataFrame({"Features": X_train.columns, "Importances":rfr.feature_importances_})
#sort in ascending order to better visualization
importance=importance.sort_values('Importances')
print(importance)
#plot the feature importances in bars
importance.plot.bar(x="Features",y="Importances",figsize=(10,5))

#model 4: MLPRegressor
from sklearn.neural_network import MLPRegressor
mlp=MLPRegressor(hidden_layer_sizes=(10,5), activation='logistic', solver='adam', learning_rate_init=0.001, random_state=18)
mlp.fit(X_train_std,y_train_std)
y_pred_train = mlp.predict(X_train_std)
y_pred_test = mlp.predict(X_test_std)
y_pred_train=scaler.inverse_transform(y_pred_train)
y_pred_test=scaler.inverse_transform(y_pred_test)
output=[]
temp=[mlp.score(X_train_std,y_train_std),mean_squared_error(y_train,y_pred_train),mean_absolute_error(y_train,y_pred_train),mlp.score(X_test_std,y_test_std),mean_squared_error(y_test,y_pred_test),mean_absolute_error(y_test,y_pred_test)]
output.append(temp)
output=pd.DataFrame(output,columns=['Train_R2','Train_MSE','Train_MAE','Test_R2','Test_MSE','Test_MAE'])
output.to_csv('MLP.csv',index=False)
plots(y_test,y_pred_test)
