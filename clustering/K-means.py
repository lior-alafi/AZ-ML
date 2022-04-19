import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

df = pd.read_csv('../data/Mall_Customers.csv')
x = df.iloc[:,1:].values
x = ColumnTransformer([('cat',OneHotEncoder(),[0])],remainder='passthrough').fit_transform(x)
print(df.head())
def identify_optimal_number_of_clusters(x):
    inertias = [] #wcss
    lim = len(x) // 2
    # lim = 12
    for num_of_clusters in range(1,lim):
        kmeans = KMeans(n_clusters=num_of_clusters, n_init=10, max_iter=300, init='k-means++')
        kmeans.fit(x)
        inertias.append(kmeans.inertia_)

    plt.title('The Elbow Method(K-means)')
    plt.xlabel('Number of clusters')
    plt.ylabel('WCSS')
    plt.plot(range(1,lim),inertias)
    plt.show()

identify_optimal_number_of_clusters(x)
kmeans = KMeans(n_clusters=5, n_init=10, max_iter=1000, init='k-means++')
clusters = kmeans.fit_predict(x)
clusters = clusters.reshape([clusters.shape[0],1])
x_mod = np.hstack((x,clusters))
print(x_mod[0])
males = x_mod[:,0] == 0


# cluster visualization
plt.scatter(x_mod[np.logical_and(males,x_mod[:,-1] == 0),3],x_mod[np.logical_and(males,x_mod[:,-1] == 0) ,4],color='red',label='cluster#1(M)',marker='v')
plt.scatter(x_mod[np.logical_and(np.logical_not(males),x_mod[:,-1] == 0),3],x_mod[np.logical_and(np.logical_not(males),x_mod[:,-1] == 0) ,4],color='red',label='cluster#1(F)')

plt.scatter(x_mod[np.logical_and(males,x_mod[:,-1] == 1),3],x_mod[np.logical_and(males,x_mod[:,-1] == 1),4],color='blue',label='cluster#2(M)',marker='v')
plt.scatter(x_mod[np.logical_and(np.logical_not(males),x_mod[:,-1] == 1),3],x_mod[np.logical_and(np.logical_not(males),x_mod[:,-1] == 1),4],color='blue',label='cluster#2(F)')

plt.scatter(x_mod[np.logical_and(males,x_mod[:,-1] == 2),3],x_mod[np.logical_and(males,x_mod[:,-1] == 2),4],color='purple',label='cluster#3(M)',marker='v')
plt.scatter(x_mod[np.logical_and(np.logical_not(males),x_mod[:,-1] == 2),3],x_mod[np.logical_and(np.logical_not(males),x_mod[:,-1] == 2),4],color='purple',label='cluster#3(F)')

plt.scatter(x_mod[np.logical_and(males,x_mod[:,-1] == 3),3],x_mod[np.logical_and(males,x_mod[:,-1] == 3),4],color='green',label='cluster#4(M)',marker='v')
plt.scatter(x_mod[np.logical_and(np.logical_not(males),x_mod[:,-1] == 3),3],x_mod[np.logical_and(np.logical_not(males),x_mod[:,-1] == 3),4],color='green',label='cluster#4(F)')

# plt.scatter(x_mod[np.logical_and(males,x_mod[:,-1] == 4),3],x_mod[np.logical_and(males,x_mod[:,-1] == 4),4],color='black',label='cluster#5(M)',marker='v')
# plt.scatter(x_mod[np.logical_and(np.logical_not(males),x_mod[:,-1] == 4),3],x_mod[np.logical_and(np.logical_not(males),x_mod[:,-1] == 4),4],color='black',label='cluster#5(F)')
#
# plt.scatter(x_mod[np.logical_and(males,x_mod[:,-1] == 5),3],x_mod[np.logical_and(males,x_mod[:,-1] == 5),4],color='cyan',label='cluster#5(M)',marker='v')
# plt.scatter(x_mod[np.logical_and(np.logical_not(males),x_mod[:,-1] == 5),3],x_mod[np.logical_and(np.logical_not(males),x_mod[:,-1] == 5),4],color='cyan',label='cluster#5(F)')
#
# plt.scatter(x_mod[np.logical_and(males,x_mod[:,-1] == 6),3],x_mod[np.logical_and(males,x_mod[:,-1] == 6),4],color='magenta',label='cluster#5(M)',marker='v')
# plt.scatter(x_mod[np.logical_and(np.logical_not(males),x_mod[:,-1] == 6),3],x_mod[np.logical_and(np.logical_not(males),x_mod[:,-1] == 6),4],color='magenta',label='cluster#5(F)')

#centroids
plt.scatter(kmeans.cluster_centers_[:,3],kmeans.cluster_centers_[:,4],marker='+',color='yellow',s=300,label='centroid')
plt.title('K Means Clustering')
plt.xlabel('Annual Income')
plt.ylabel('Spending Score')
plt.legend()
plt.show()