from xgboost import XGBClassifier
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('../data/xgboostdata.csv')
X,y = df.iloc[:,1:-1].values,df.iloc[:,-1].values
x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.2)
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

xgboostclassifer = XGBClassifier()
xgboostclassifer.fit(x_train,y_train)
y_hat = xgboostclassifer.predict(x_test)
#
from sklearn.metrics import accuracy_score,f1_score
#
print(accuracy_score(y_test,y_hat))