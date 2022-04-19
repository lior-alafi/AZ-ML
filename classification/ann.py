import sys

import tensorflow as tf

from sklearn.preprocessing import StandardScaler,OneHotEncoder,LabelEncoder
from sklearn.compose import ColumnTransformer
print(tf.__version__)
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras import layers
import numpy as np
import pandas as pd
np.set_printoptions(threshold=sys.maxsize)
df = pd.read_csv('../data/Churn_Modelling.csv')
X = df.iloc[:,3:-1].values
y = df.iloc[:,-1].values
ct = ColumnTransformer(transformers =[('c',OneHotEncoder(),[1])],remainder='passthrough')
X = np.array(ct.fit_transform(X))

X[:,4] = LabelEncoder().fit_transform(X[:,4])

x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.20)
#
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

model = Sequential(
    [

        layers.Dense(6, activation='relu'),
        layers.Dense(6, activation='relu'),
        layers.Dense(1, activation='sigmoid', name="sigm"),
    ]
)
from keras.optimizers import adam_v2
model.compile(optimizer=adam_v2.Adam(0.001),loss='binary_crossentropy',metrics=['accuracy',tf.keras.metrics.AUC()])

model.fit(x_train,y_train,batch_size=32,epochs=180)
y_hat = model.predict(x_test)
y_hat = (y_hat > 0.5)
print(np.concatenate((y_hat.reshape(len(y_hat),1), y_test.reshape(len(y_test),1)),1))

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_hat)
print(cm)
accuracy = np.trace(cm)/np.sum(cm)
percision = cm[1,1]/np.sum(cm[:,1])
recall = cm[1,1]/np.sum(cm[1,:])
print(f'acc: {accuracy} percision: {percision} recall: {recall} f1: {2*percision*recall/(percision+recall)}')