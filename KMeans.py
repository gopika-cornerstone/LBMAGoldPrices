import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import r2_score

# Load the dataset
df = pd.read_csv('/Users/gopikasriram/Desktop/LBMAGold/clean.csv', index_col=0)

# Preprocess the data
X = df[['USD (Average)', 'GBP (Average)', 'EURO (Average)']].values

# Determine the optimal number of clusters using the elbow method
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
    
# Plot the elbow graph to determine the optimal number of clusters
import matplotlib.pyplot as plt
plt.plot(range(1, 11), wcss)
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

# Perform k-means clustering
kmeans = KMeans(n_clusters=3, init='k-means++', max_iter=300, n_init=10, random_state=0)
y_kmeans = kmeans.fit_predict(X)

# Calculate the R-squared score
r_squared = r2_score(X, kmeans.cluster_centers_[y_kmeans])

# Print the results
print('Cluster Centers:')
print(kmeans.cluster_centers_)
print('\nCluster Labels:')
print(y_kmeans)
print('\nR-Squared Score:')
print(r_squared)
