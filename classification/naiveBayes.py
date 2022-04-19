import pandas as pd
import numpy as np
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay,RocCurveDisplay

df = pd.read_csv('../data/Social_Network_Ads.csv')
x,y = df.iloc[:,:-1].values,df.iloc[:,-1].values
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2)
xscaler = StandardScaler()
x_train = xscaler.fit_transform(x_train,y_train)

naiveBayesClassifier = GaussianNB()
naiveBayesClassifier.fit(x_train,y_train)
y_hat = naiveBayesClassifier.predict(xscaler.transform(x_test))
print(naiveBayesClassifier.predict(xscaler.transform([[46,204000]])))
ConfusionMatrixDisplay.from_estimator(naiveBayesClassifier,xscaler.transform(x_test),y_test)
plt.show()
RocCurveDisplay.from_estimator(naiveBayesClassifier,xscaler.transform(x_test),y_test)
plt.show()


xr = np.arange(np.min(x_test[:,0])-10,np.max(x_test[:,0])+10,0.5)
yr = np.arange(np.min(x_test[:,1])-1000,np.max(x_test[:,1])+1000,0.5)
x1,x2 = np.meshgrid(xr,yr)
grid = np.array([x1.ravel(),x2.ravel()]).T
grid = xscaler.transform(grid)
pred_grid = naiveBayesClassifier.predict(grid)

plt.contourf(x1,x2,pred_grid.reshape(x1.shape),  alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(x1.min(), x1.max())
plt.ylim(x2.min(), x2.max())
for i, j in enumerate(np.unique(y_test)):
    plt.scatter(x_test[y_test == j, 0], x_test[y_test == j, 1], c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Naive Bayes (radial kernel) (Test set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()
