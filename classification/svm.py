from matplotlib.colors import ListedColormap
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import ConfusionMatrixDisplay,RocCurveDisplay
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

df = pd.read_csv('../data/Social_Network_Ads.csv')
x,y = df.iloc[:,:-1].values,df.iloc[:,-1].values
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2)
xscaler = StandardScaler()
x_train = xscaler.fit_transform(x_train)

svc = SVC(kernel='rbf',tol=1e-4,C=1.0)
svc.fit(x_train,y_train)
ConfusionMatrixDisplay.from_estimator(svc,xscaler.transform(x_test),y_test)
plt.show()
RocCurveDisplay.from_estimator(svc,xscaler.transform(x_test),y_test)
plt.show()



xr = np.arange(np.min(x_test[:,0])-10, np.max(x_test[:,0]) + 10, 0.25)
yr = np.arange(np.min(x_test[:,1])-1000, np.max(x_test[:,1]) + 1000, 0.25)
x1,x2 = np.meshgrid(xr,yr)
grid = xscaler.transform(np.array([x1.ravel(),x2.ravel()]).T)
pred_grid = svc.predict(grid)
plt.contourf(x1,x2,pred_grid.reshape(x1.shape),  alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(x1.min(), x1.max())
plt.ylim(x2.min(), x2.max())
for i, j in enumerate(np.unique(y_test)):
    plt.scatter(x_test[y_test == j, 0], x_test[y_test == j, 1], c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('SVM(rbf kernel) (Test set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()


