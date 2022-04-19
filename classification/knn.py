from matplotlib.colors import ListedColormap
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay,RocCurveDisplay
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

df = pd.read_csv('../data/Social_Network_Ads.csv')
x,y = df.iloc[:,:-1].values,df.iloc[:,-1].values
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2)
xscaler = StandardScaler()
x_train = xscaler.fit_transform(x_train)
#weight can be also "distance"
knnClassifier = KNeighborsClassifier(n_neighbors=5,weights='uniform',metric='minkowski',p=2)
knnClassifier.fit(x_train,y_train)
y_hat = knnClassifier.predict(xscaler.transform(x_test))
print(f'knn score: {knnClassifier.score(xscaler.transform(x_test),y_test)}')
ConfusionMatrixDisplay.from_estimator(knnClassifier,xscaler.transform(x_test),y_test)
plt.show()
RocCurveDisplay.from_estimator(knnClassifier,xscaler.transform(x_test),y_test)
plt.show()


#visualizing
x1 = np.arange(start=np.min(x_test[:,0])-10 ,stop=np.max(x_test[:,0]) +10,step=1)
x2 = np.arange(start=np.min(x_test[:,1]) -1000 ,stop=np.max(x_test[:,1]) +1000,step=1)
X1,X2 = np.meshgrid(x1,x2)

grid = xscaler.transform(np.array([X1.ravel(), X2.ravel()]).T)
plt.contourf(X1, X2, knnClassifier.predict(grid).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_test)):
    plt.scatter(x_test[y_test == j, 0], x_test[y_test == j, 1], c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('KNN (Test set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()

#
# # Visualising the Test set results
# from matplotlib.colors import ListedColormap
# X_set, y_set = x_test, y_test
# X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 10, stop = X_set[:, 0].max() + 10, step = 1),
#                      np.arange(start = X_set[:, 1].min() - 1000, stop = X_set[:, 1].max() + 1000, step = 1))
# plt.contourf(X1, X2, knnClassifier.predict(xscaler.transform(np.array([X1.ravel(), X2.ravel()]).T)).reshape(X1.shape),
#              alpha = 0.75, cmap = ListedColormap(('red', 'green')))
# plt.xlim(X1.min(), X1.max())
# plt.ylim(X2.min(), X2.max())
# for i, j in enumerate(np.unique(y_set)):
#     plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1], c = ListedColormap(('red', 'green'))(i), label = j)
# plt.title('K-NN (Test set)')
# plt.xlabel('Age')
# plt.ylabel('Estimated Salary')
# plt.legend()
# plt.show()