#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 20 20:25:20 2018

@author: cad
"""
import numpy as np
from sklearn.cluster import KMeans
import pickle
import sys
import pdb
import matplotlib.pyplot as plt

# relevant variables
num_clusters = int(sys.argv[1])
category = sys.argv[2]
print('num_clusters: ', num_clusters)
kmeans_file = 'data/kmeans_' + str(num_clusters) + '_' + category + '.pkl'

azimuths = []
elevations = []

f = open('/home/cad/disk/linux/RenderForCNN-master/train/angle_class.txt', 'r')
for line in f.readlines():
	elevations.append(line.split()[1])
	azimuths.append(line.split()[2])
#pdb.set_trace()

ydata = []
f = open('/media/cad/0C66F84266F82DD8/vp_lmdb/' + category + '/train.txt','r')
for line in f.readlines():
	viewpoint = int(line.split()[1])
	azimuth, elevation = azimuths[viewpoint], elevations[viewpoint]
	angle = np.array([azimuth, elevation])
	ydata.append(angle)
ydata = np.stack(ydata)
print('\nData size: ', ydata.shape)

kmeans = KMeans(num_clusters, verbose=1, n_jobs=10)
kmeans.fit(ydata)
print(kmeans.cluster_centers_)

# save output
fid = open(kmeans_file, 'wb')
pickle.dump(kmeans, fid)

del kmeans

# load and check
kmeans = pickle.load(open(kmeans_file, 'rb'))
print(kmeans.cluster_centers_)

azimuths = []
elevations = []
for idx in kmeans.cluster_centers_:
	azimuths.append(idx[0])
	elevations.append(idx[1])	

pdb.set_trace()
plt.scatter(azimuths, elevations)
plt.show()
