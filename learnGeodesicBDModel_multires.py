# -*- coding: utf-8 -*-
"""
Function that learns feature model + 3layer pose models x 12 object categories
in an end-to-end manner by minimizing the mean squared error for axis-angle representation
"""

import torch
from torch import nn, optim
from torch.autograd import Variable
from torch.utils.data import DataLoader

from dataGenerators import MultibinImages, Pascal3dAll, my_collate
from featureModels import resnet_model
from axisAngle import get_error, geodesic_loss
from binDeltaModels import bin_3layer, res_2layer
from binDeltaLosses import loss_m0, loss_m1
from helperFunctions import classes

import numpy as np
import scipy.io as spio
import gc
import os
import time
import progressbar
import sys
import pickle

if len(sys.argv) > 1:
	os.environ['CUDA_VISIBLE_DEVICES'] = sys.argv[1]

# relevant paths
augmented_path = 'data/augmented2/'
pascal3d_path = 'data/original'

# save stuff here
save_str = 'resnet50_2layer_multires_m1_1'
results_file = os.path.join('results', save_str)
model_file = os.path.join('models', save_str + '.tar')
plots_file = os.path.join('plots', save_str)

# kmeans data
kmeans_file = 'data/kmeans_dictionary_axis_angle_16.pkl'
kmeans = pickle.load(open(kmeans_file, 'rb'))
kmeans_dict = kmeans.cluster_centers_
num_clusters = kmeans.n_clusters

# relevant variables
ndim = 3
num_workers = 4
N0 = 2048
N1 = 1000
N2 = 500
N3 = 100
num_classes = len(classes)
init_lr = 0.0001
num_epochs = 3

problem_type = 'm1'
criterion1 = loss_m0(1.0)
criterion2 = loss_m1(1.0, kmeans_file, geodesic_loss().cuda())

# DATA
# datasets
real_data = MultibinImages(augmented_path, 'real', problem_type, kmeans_file)
print(real_data)
test_data = Pascal3dAll(pascal3d_path, 'test')
# setup data loaders
real_loader = DataLoader(real_data, batch_size=4, shuffle=True, num_workers=num_workers, pin_memory=True, collate_fn=my_collate)
test_loader = DataLoader(test_data, batch_size=8, collate_fn=my_collate)
print('Real: {0} \t Test: {1}'.format(len(real_loader),  len(test_loader)))


# MODEL
# my model for pose estimation: feature model + 1layer pose model x 12
class my_model(nn.Module):
	def __init__(self):
		super().__init__()
		self.num_classes = num_classes
		self.num_clusters = num_clusters
		self.feature_model = resnet_model('resnet50', 'layer4').cuda()
		self.bin_models = nn.Sequential(bin_3layer(N0, N1, N2, num_clusters)).cuda()
		self.res_models = nn.ModuleList([res_2layer(N0, N3, ndim) for i in range(self.num_clusters)]).cuda()	
		#print(self.bin_models)

	def forward(self, x, class_label):
		x = self.feature_model(x)
		y1 = torch.stack([self.bin_models[i](x) for i in range(self.num_classes)]).permute(1, 2, 0)
		y2 = torch.stack([self.res_models[i](x) for i in range(self.num_classes * self.num_clusters)])
		y2 = y2.view(self.num_classes, self.num_clusters, -1, ndim).permute(1, 2, 3, 0)
		class_label = torch.zeros(class_label.size(0), self.num_classes).scatter_(1, class_label.data.cpu(), 1.0)
		class_label = Variable(class_label.unsqueeze(2).cuda())
		y1 = torch.squeeze(torch.bmm(y1, class_label), 2)
		y2 = torch.squeeze(torch.matmul(y2, class_label), 3)
		_, pose_label = torch.max(y1, dim=1, keepdim=True)
		pose_label = torch.zeros(pose_label.size(0), self.num_clusters).scatter_(1, pose_label.data.cpu(), 1.0)
		pose_label = Variable(pose_label.unsqueeze(2).cuda())
		y2 = torch.squeeze(torch.bmm(y2.permute(1, 2, 0), pose_label), 2)
		del x, class_label, pose_label
		return [y1, y2]


# my_model
model = my_model()
#print(model)
# loss and optimizer
optimizer = optim.Adam(model.parameters(), lr=init_lr)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)
train_loss = []
train_loss_sum = 0.0
train_samples = 0


# OPTIMIZATION functions
def training_m0(save_loss=False):
	global train_loss_sum
	global train_samples
	model.train()
	bar = progressbar.ProgressBar(maxval=len(real_loader))
	bar.start()
	for i, sample_real in enumerate(real_loader):
		# forward steps
		xdata_real = Variable(sample_real['xdata'].cuda())
		label_real = Variable(sample_real['label'].cuda())
		ydata_real = [Variable(sample_real['ydata_bin'].cuda()), Variable(sample_real['ydata_res'].cuda())]
		output_real = model(xdata_real, label_real)
		loss_real = criterion1(output_real, ydata_real)
		loss = loss_real
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
		# store
		bar.update(i)
		train_loss_sum += (loss_real.data[0] * xdata_real.size(0))
		train_samples += (xdata_real.size(0))
		if i % 1000 == 0 and save_loss:
			train_loss.append(train_loss_sum / train_samples)
		# cleanup
		del xdata_real, label_real,  ydata_real
		del output_real, loss_real, sample_real, loss
		gc.collect()
	real_loader.dataset.shuffle_images()


def training_m1(save_loss=False):
	global train_loss_sum
	global train_samples
	model.train()
	bar = progressbar.ProgressBar(maxval=len(real_loader))
	bar.start()
	for i, sample_real in enumerate(real_loader):
		# forward steps
		xdata_real = Variable(sample_real['xdata'].cuda())
		label_real = Variable(sample_real['label'].cuda())
		ydata_real = [Variable(sample_real['ydata_bin'].cuda()), Variable(sample_real['ydata'].cuda())]
		output_real = model(xdata_real, label_real)
		loss_real = criterion2(output_real, ydata_real)
		loss = loss_real
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
		# store
		bar.update(i)
		train_loss_sum += (loss_real.data[0] * xdata_real.size(0))
		train_samples += (xdata_real.size(0) )
		if i % 1000 == 0 and save_loss:
			train_loss.append(train_loss_sum / train_samples)
		# cleanup
		del xdata_real, label_real, ydata_real
		del output_real, loss_real, sample_real, loss
		gc.collect()
	real_loader.dataset.shuffle_images()


def testing():
	model.eval()
	bar = progressbar.ProgressBar(maxval=len(test_loader))
	ypred = []
	ytrue = []
	labels = []
	bar.start()
	for i, sample in enumerate(test_loader):
		xdata = Variable(sample['xdata'].cuda())
		label = Variable(sample['label'].cuda())
		output = model(xdata, label)
		ypred_bin = np.argmax(output[0].data.cpu().numpy(), axis=1)
		ypred_res = output[1].data.cpu().numpy()
		ypred.append(kmeans_dict[ypred_bin, :] + ypred_res)
		ytrue.append(sample['ydata'].numpy())
		labels.append(sample['label'].numpy())
		bar.update(i)
		del xdata, label, output, sample
		gc.collect()
	ypred = np.concatenate(ypred)
	ytrue = np.concatenate(ytrue)
	labels = np.concatenate(labels)
	model.train()
	return ytrue, ypred, labels


def save_checkpoint(filename):
	torch.save(model.state_dict(), filename)


# minimize only the bin part
training_m0(True)
ytest, yhat_test, _ = testing()
get_error(ytest, yhat_test)
train_init = train_loss

train_loss = []
train_loss_sum = 0.0
train_samples = 0
for epoch in range(num_epochs):
	tic = time.time()
	scheduler.step()
	# training step
	training_m1(True)
	# save model at end of epoch
	save_checkpoint(model_file)
	# time and output
	toc = time.time() - tic
	print('Epoch: {0} done in time {1}s'.format(epoch, toc))
	# cleanup
	gc.collect()
# save plots
#spio.savemat(plots_file, {'train_loss': train_loss, 'train_init': train_init})

# evaluate the model
ytest, yhat_test, test_labels = testing()
get_error(ytest, yhat_test)
spio.savemat(results_file, {'ytest': ytest, 'yhat_test': yhat_test, 'test_labels': test_labels})
