import glob
import numpy as np
import matplotlib.pyplot as plt
import sklearn.cluster



images = glob.glob("./imagenes/*.png")

data=[]
for i in range(len(images)):
    d=np.float_(plt.imread(images[i]).flatten())
    data.append(d)

data = np.array(data)


# n_cluster_array = 
n_clusters = np.arange(1,21,1)
inertia_array = []

for nc in n_clusters:
    k_means = sklearn.cluster.KMeans( n_clusters=nc )
    k_means.fit(data)

    # calculo a cual cluster pertenece cada pixel
    cluster = k_means.predict(data)

    # asigno a cada pixel el lugar del centro de su cluster
    data_centered = data.copy()
    for i in range(nc):
        ii = cluster==i
        data_centered[ii,:] = np.int_(k_means.cluster_centers_[i])
    
    inertia_array.append( k_means.inertia_ )

inertia_array = np.array( inertia_array )


plt.figure()
plt.scatter(n_clusters, inertia_array)
plt.savefig('inercia.png')


h = n_clusters[1]-n_clusters[0]
inertia_slope = (inertia_array[1:] - inertia_array[0:-1])/h
plt.scatter(n_clusters[0:-1], inertia_slope)

n_opt = 4

k_means = sklearn.cluster.KMeans( n_clusters=n_opt )
k_means.fit(data)

# asigno a cada pixel el lugar del centro de su cluster
data_centered = data.copy()
cluster_centers =  k_means.cluster_centers_ 


min_distance = []
for n in range(n_opt):
    distance = []
    for d in range(len(data)):
        distance.append(np.linalg.norm(cluster_centers[n] - data[d]))
    min_distance.append(np.argsort(distance)[:5])
    
min_distance = np.array(min_distance).flatten()


plt.figure(figsize=(15,10))
for i in range(20):
    plt.subplot(4,5,i+1)
    data = plt.imread(images[min_distance[i]])
    plt.imshow(data) 
plt.savefig('ejemplo_clases.png')