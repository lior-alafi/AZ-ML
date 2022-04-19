import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap,Colormap


df = pd.read_csv('../data/Social_Network_Ads.csv')
x,y = df.iloc[:,:-1].values,df.iloc[:,-1].values
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2)
xscaler = StandardScaler()
x_train = xscaler.fit_transform(x_train)
logistic = LogisticRegression(penalty='l2',tol=1e-4)
logistic.fit(x_train,y_train)
print(f'score: {logistic.score(xscaler.transform(x_test),y_test)}')
my_data = [[35,18000]]
print(logistic.predict(xscaler.transform(my_data)))


from sklearn.metrics import confusion_matrix, f1_score, RocCurveDisplay, \
    ConfusionMatrixDisplay,recall_score,precision_score

confusionMatrix = confusion_matrix(y_test,logistic.predict(xscaler.transform(x_test)))
print(confusionMatrix)
ConfusionMatrixDisplay.from_estimator(logistic,xscaler.transform(x_test),y_test)
plt.show()
precision = confusionMatrix[1,1]/ (confusionMatrix[0,1]+confusionMatrix[1,1]) #true_positive/(true_positive + false_positive)
recall = confusionMatrix[1,1]/ (confusionMatrix[1,0]+confusionMatrix[1,1]) #true_positive/(true_positive + false_negative)
print(f'my precision: {precision} true: {precision_score(y_test,logistic.predict(xscaler.transform(x_test)))}')
print(f'my recall: {recall} {recall_score(y_test,logistic.predict(xscaler.transform(x_test)))}')
print(f1_score(y_test,logistic.predict(xscaler.transform(x_test))))
accuracy = (confusionMatrix[0,0]+confusionMatrix[1,1]) / (np.sum(confusionMatrix.ravel()))
print(accuracy)
RocCurveDisplay.from_estimator(estimator=logistic,X=xscaler.transform(x_test),y=y_test)
plt.show()


#visualizing
x1 = np.arange(start=np.min(x_test[:,0])-10 ,stop=np.max(x_test[:,0]) +10,step=0.25)
x2 = np.arange(start=np.min(x_test[:,1]) -1000 ,stop=np.max(x_test[:,1]) +1000,step=0.25)
X1,X2 = np.meshgrid(x1,x2)

grid = xscaler.transform(np.array([X1.ravel(), X2.ravel()]).T)
plt.contourf(X1, X2, logistic.predict(grid).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_test)):
    plt.scatter(x_test[y_test == j, 0], x_test[y_test == j, 1], c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Logistic Regression (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()


