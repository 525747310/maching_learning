# -*- coding: utf-8 -*-

from skimage import io
#KMeans包
from sklearn.cluster import KMeans
import numpy as np

image = io.imread('test2.jpg')
io.imshow(image)
io.show()

rows = image.shape[0]
cols = image.shape[1]

print(rows)
print(cols)

#把图像进行操作,每个点r，g，b（1*3）为一个样本
image = image.reshape(image.shape[0] * image.shape[1], 3)
#把范围聚类到128个簇中
kmeans = KMeans(n_clusters=128, n_init=10, max_iter=200)   #max_iter：最大迭代次数
kmeans.fit(image)

#kmeans.cluster_centers_:聚类的中心点
clusters = np.asarray(kmeans.cluster_centers_, dtype=np.uint8)
print(clusters)
labels = np.asarray(kmeans.labels_, dtype=np.uint8)
print('labels.shape:',labels.shape)
labels = labels.reshape(rows, cols);
print('labels.shape:',labels.shape)

print(clusters.shape)
np.save('codebook_test.npy', clusters)
io.imsave('compressed_test.jpg', labels)

image = io.imread('compressed_test.jpg')
io.imshow(image)
io.show()