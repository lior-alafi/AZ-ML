import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram,linkage


def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)


df = pd.read_csv('../data/Mall_Customers.csv')
x = df.iloc[:,1:].values
print(df.head())

x = ColumnTransformer([('cat',OneHotEncoder(),[0])],remainder='passthrough').fit_transform(x)
print(df.head())
dendogramplot = dendrogram(linkage(x,method='ward'))
plt.title("dendogram with categorical data")
plt.xlabel('customers')
plt.ylabel('distances')
plt.show()

# agglomerativeHierarchialCluster = AgglomerativeClustering(n_clusters=None,distance_threshold=0)
# agglomerativeHierarchialCluster.fit(x)
# # plot_dendrogram(agglomerativeHierarchialCluster)


agglomerativeHierarchialCluster = AgglomerativeClustering(n_clusters=5,affinity='euclidean',linkage='ward')
clusters  = agglomerativeHierarchialCluster.fit_predict(x)
print(clusters)
# plot_dendrogram(agglomerativeHierarchialCluster)
plt.show()
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

plt.scatter(x_mod[np.logical_and(males,x_mod[:,-1] == 4),3],x_mod[np.logical_and(males,x_mod[:,-1] == 4),4],color='black',label='cluster#5(M)',marker='v')
plt.scatter(x_mod[np.logical_and(np.logical_not(males),x_mod[:,-1] == 4),3],x_mod[np.logical_and(np.logical_not(males),x_mod[:,-1] == 4),4],color='black',label='cluster#5(F)')

# plt.scatter(x_mod[np.logical_and(males,x_mod[:,-1] == 5),3],x_mod[np.logical_and(males,x_mod[:,-1] == 5),4],color='cyan',label='cluster#5(M)',marker='v')
# plt.scatter(x_mod[np.logical_and(np.logical_not(males),x_mod[:,-1] == 5),3],x_mod[np.logical_and(np.logical_not(males),x_mod[:,-1] == 5),4],color='cyan',label='cluster#5(F)')
#
# plt.scatter(x_mod[np.logical_and(males,x_mod[:,-1] == 6),3],x_mod[np.logical_and(males,x_mod[:,-1] == 6),4],color='magenta',label='cluster#5(M)',marker='v')
# plt.scatter(x_mod[np.logical_and(np.logical_not(males),x_mod[:,-1] == 6),3],x_mod[np.logical_and(np.logical_not(males),x_mod[:,-1] == 6),4],color='magenta',label='cluster#5(F)')
plt.title('Agglomerative Clustering')
plt.xlabel('Annual Income')
plt.ylabel('Spending Score')
plt.legend()
plt.show()