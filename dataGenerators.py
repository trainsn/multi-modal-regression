# -*- coding: utf-8 -*-
"""
Data generators to load images
"""

import torch
from torch.utils.data import Dataset
from torchvision import transforms

from helperFunctions import parse_name, rotation_matrix, classes, eps
from axisAngle import get_y, get_R
from quaternion import get_y as get_quaternion

from PIL import Image
import numpy as np
import scipy.io as spio
from scipy.spatial.distance import cdist
import os
import pickle
import pdb

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
preprocess_render = transforms.Compose([transforms.Resize([224, 224]), transforms.ToTensor(), normalize])
preprocess_real = transforms.Compose([transforms.ToTensor(), normalize])


class ImagesAll(Dataset):
	def __init__(self, db_path, db_type, ydata_type='axis_angle'):
		self.db_path = db_path
		self.classes = classes
		self.num_classes = len(self.classes)
		self.db_type = db_type
		self.ydata_type = ydata_type
		self.list_image_names = []
		for i in range(1):
		    for filename in os.listdir(os.path.join(self.db_path, self.classes[i])):
			    	#print(filename)
			    	self.list_image_names.append(filename)
#		for i in range(self.num_classes):
#			tmp = spio.loadmat(os.path.join(self.db_path, self.classes[i] + '_info'), squeeze_me=True)
#			#print(tmp)
#			image_names = tmp['imagenet_train']
#			self.list_image_names.append(image_names)
		self.num_images = np.array([len(self.list_image_names[i]) for i in range(self.num_classes)])
		normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
		self.preprocess = transforms.Compose([transforms.Resize([224, 224]), transforms.ToTensor(), normalize])
		self.image_names = self.list_image_names
		#print(self.image_names)

	def __len__(self):
		return np.amax(self.num_images)

	def __getitem__(self, idx):
		# return sample with xdata, ydata, label
		xdata, ydata, label = [], [], []
		for i in range(1):
    
			image_name = self.image_names[i]
			label.append(i*torch.ones(1).long())   
			# read image
			#print(self.image_names)
			img_pil = Image.open(os.path.join(self.db_path, self.classes[i], image_name))
			xdata.append(self.preprocess(img_pil))
			# parse image name to get correponding target
			#pdb.set_trace()
			_, _, az, el, ct, _ = parse_name(image_name)
			if self.db_type == 'real':
				R = rotation_matrix(az, el, ct)
			elif self.db_type == 'render':
				R = rotation_matrix(az, el, -ct)
			else:
				raise NameError('Unknown db_type passed')
			if self.ydata_type == 'axis_angle':
				tmpy = get_y(R)
			elif self.ydata_type == 'quaternion':
				tmpy = get_quaternion(R)
			else:
				raise NameError('Uknown ydata_type passed')
			ydata.append(torch.from_numpy(tmpy).float())
		xdata = torch.stack(xdata)
		ydata = torch.stack(ydata)
		label = torch.stack(label)
		sample = {'xdata': xdata, 'ydata': ydata, 'label': label}
		return sample

	def shuffle_images(self):
		#self.image_names = [np.random.permutation(self.list_image_names[i]) for i in range(self.num_classes)]
		pass


class Pascal3dAll(Dataset):
	def __init__(self, db_path, db_type, ydata_type='axis_angle'):
		super().__init__()
		self.classes = classes
		self.num_classes = len(self.classes)
		self.db_path = db_path
		self.db_type = db_type
		self.ydata_type = ydata_type
		self.list_image_names = []
		self.labels = []
		for i in range(self.num_classes):
			tmp = spio.loadmat(os.path.join(self.db_path, self.classes[i] + '_info'), squeeze_me=True)
			if self.db_type == 'val':
				self.list_image_names.append(tmp['pascal_train'])
				self.labels.append(i * np.ones(len(tmp['pascal_train']), dtype='int'))
			else:
				self.list_image_names.append(tmp['pascal_val'])
				self.labels.append(i * np.ones(len(tmp['pascal_val']), dtype='int'))
		self.image_names = np.concatenate(self.list_image_names)
		self.labels = np.concatenate(self.labels)

	def __len__(self):
		return len(self.image_names)

	def __getitem__(self, idx):
		image_name = self.image_names[idx]
		image_label = self.labels[idx]
		image_path = os.path.join(self.db_path, self.classes[image_label], image_name)
		tmp = spio.loadmat(image_path, verify_compressed_data_integrity=False)
		xdata = tmp['xdata']
		if self.ydata_type == 'axis_angle':
			ydata = tmp['ydata']
		elif self.ydata_type == 'quaternion':
			angle = np.linalg.norm(tmp['ydata'], 2, 1, True)
			axis = tmp['ydata'] / np.maximum(eps, angle)
			ydata = np.concatenate([np.cos(angle/2.0), np.sin(angle/2.0) * axis], axis=1)
		else:
			raise NameError('Uknown ydata_type passed')
		label = image_label * np.ones((ydata.shape[0], 1))
		# get torch tensors from this data
		xdata = torch.stack([preprocess_real(xdata[i]) for i in range(xdata.shape[0])]).float()
		ydata = torch.from_numpy(ydata).float()
		label = torch.from_numpy(label).long()
		sample = {'xdata': xdata, 'ydata': ydata, 'label': label}
		return sample


def my_collate(list_samples):
	my_keys = list_samples[0].keys()
	new_sample = {}
	for key in my_keys:
		new_sample[key] = torch.cat([sample[key] for sample in list_samples])
	return new_sample


class MultibinImages(ImagesAll):
	def __init__(self, db_path, db_type, problem_type, kmeans_file):
		# initialize the renderedImages dataset first
		super().__init__(db_path, db_type)
		self.problem_type = problem_type
		# add the kmeans part
		self.kmeans = pickle.load(open(kmeans_file, 'rb'))
		self.num_clusters = self.kmeans.n_clusters
		#pdb.set_trace()
		if self.problem_type == 'm2':
			self.key_rotations = [get_R(y) for y in self.kmeans.cluster_centers_]

	def __len__(self):
		return np.amax(self.num_images)

	def __getitem__(self, idx):
		# run the item handler of the renderedImages dataset
		sample = super().__getitem__(idx)
		# update the ydata target using kmeans dictionary
		ydata = sample['ydata'].numpy()
		# bin part
		if self.problem_type == 'm3':
			ydata_bin = np.exp(-10.0*cdist(ydata, self.kmeans.cluster_centers_, 'sqeuclidean'))
			ydata_bin = ydata_bin/np.sum(ydata_bin, axis=1, keepdims=True)
			sample['ydata_bin'] = torch.from_numpy(ydata_bin).float()
		else:
			ydata_bin = self.kmeans.predict(ydata)
			sample['ydata_bin'] = torch.from_numpy(ydata_bin).long()
		# residual part
		if self.problem_type == 'm2':
			ydata_res = get_residuals(ydata, self.key_rotations)
		elif self.problem_type == 'm3':
			ydata_res = ydata - np.dot(ydata_bin, self.kmeans.cluster_centers_)     # need to think more about m4
		else:
			ydata_res = ydata - self.kmeans.cluster_centers_[ydata_bin, :]
		sample['ydata_res'] = torch.from_numpy(ydata_res).float()
		#print(sample)
		return sample
	
# use PIL Image to read image
def default_loader(path):
    try:
        img = Image.open(path)
        return img.convert('RGB')
    except:
        print("Cannot read image: {}".format(path))
		
class ViewpointImages(Dataset):
	def __init__(self, img_path, txt_path, kmeans_file, dataset = '', data_transforms=None, loader = default_loader):
		with open(txt_path) as input_file:
			lines = input_file.readlines()
			self.img_name = [os.path.join(line.strip().split(' ')[0]) for line in lines]
			self.img_label = [int(line.strip().split(' ')[-1]) for line in lines]
		self.data_transforms = data_transforms
		self.dataset = dataset
		self.loader = loader
		self.kmeans = pickle.load(open(kmeans_file, 'rb'))
		self.num_clusters = self.kmeans.n_clusters
		
		self.azimuths = []
		self.elevations = []
		f = open('/home/cad/disk/linux/RenderForCNN-master/train/angle_class.txt', 'r')
		for line in f.readlines():
			self.elevations.append(float(line.split()[1]))
			self.azimuths.append(float(line.split()[2]))
		

	def __len__(self):
		return len(self.img_name)

	def __getitem__(self, item):
		img_name = self.img_name[item]
		label = self.img_label[item]
		xdata = self.loader(img_name)
		ydata = []
		if self.data_transforms is not None:
			try:
				xdata = self.data_transforms[self.dataset](xdata)
			except:
				print("Cannot transform image: {}".format(img_name))

		ydata = np.array([[self.azimuths[label], self.elevations[label]]])	
		#binpart 
		ydata_bin = self.kmeans.predict(ydata)
		# residual part	
		ydata_res = ydata - self.kmeans.cluster_centers_[ydata_bin, :]
		#print(ydata, ydata_bin, ydata_res)
		sample = {'xdata': xdata, 'ydata': ydata}
		sample['ydata_bin'] = torch.from_numpy(ydata_bin).long()
		sample['ydata_res'] = torch.from_numpy(ydata_res).float()
		#pdb.set_trace()
		
		return sample


def get_residuals(ydata, key_rotations):
	ydata_res = np.zeros((ydata.shape[0], len(key_rotations), 3))
	for i in range(ydata.shape[0]):
		for j in range(len(key_rotations)):
			ydata_res[i, j, :] = get_y(np.dot(key_rotations[j].T, get_R(ydata[i])))
	return ydata_res
