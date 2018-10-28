#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 20 19:45:58 2018

@author: cad
"""

import torch
from torch import nn, optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import  models, transforms

from dataGenerators import ViewpointImages, my_collate
from featureModels import resnet_model
from axisAngle import get_error, viewpoint_geodesic_loss
from binDeltaModels import bin_3layer, res_2layer
from binDeltaLosses import loss_m0, loss_m1
from helperFunctions import classes

import numpy as np
import os
import time
import pickle
import pdb
import math
import sys


category = sys.argv[1]
# save stuff here
save_str = 'resnet50_' + category 
results_file = os.path.join('results', save_str)
model_file = os.path.join('models', save_str + '.tar')
plots_file = os.path.join('plots', save_str)

# kmeans data
kmeans_file = 'data/kmeans_24_' + category + '.pkl'
kmeans = pickle.load(open(kmeans_file, 'rb'))
kmeans_dict = kmeans.cluster_centers_
num_clusters = kmeans.n_clusters

# relevant variables
ndim = 2
num_workers = 4
N0 = 2048
N1 = 1000
N2 = 500
N3 = 100
init_lr = 0.0001
num_epochs = 5

batch_size = {'train':48, 'val':24}
eps = 1e-6

problem_type = 'm1'
criterion1 = loss_m0(1.0)
criterion2 = loss_m1(10.0, kmeans_file, viewpoint_geodesic_loss().cuda())

data_transforms = {
		'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            #transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }


real_data = {x: ViewpointImages(img_path='/ImagePath',
                                    txt_path=('/media/cad/0C66F84266F82DD8/vp_lmdb/' + category + '/' + x + '.txt'),
								kmeans_file=kmeans_file,
                                    data_transforms=data_transforms,
                                    dataset=x) for x in ['train', 'val']}

# wrap your data and label into Tensor
dataloaders = {x: torch.utils.data.DataLoader(real_data[x],
                                                 batch_size=batch_size[x],
                                                 shuffle=True) for x in ['train', 'val']}
	
dataset_sizes = {x: len(real_data[x]) for x in ['train', 'val']}

class my_model(nn.Module):
	def __init__(self):
		super().__init__()
		self.num_clusters = num_clusters
		self.feature_model = resnet_model('resnet50', 'layer4').cuda()
		self.bin_models = nn.Sequential(bin_3layer(N0, N1, N2, num_clusters)).cuda()
		self.res_models = nn.ModuleList([res_2layer(N0, N3, ndim) for i in range(self.num_clusters)]).cuda()
		
	def forward(self, x):
		x = self.feature_model(x)
		y1 = self.bin_models(x)
		y2 = torch.stack([self.res_models[i](x) for i in range(self.num_clusters)])

		_, pose_label = torch.max(y1, dim=1, keepdim=True)
		pose_label = torch.zeros(pose_label.size(0), self.num_clusters).scatter_(1, pose_label.data.cpu(), 1.0)
		pose_label = Variable(pose_label.unsqueeze(2).cuda())
		y2 = torch.squeeze(torch.bmm(y2.permute(1, 2, 0), pose_label), 2)
		return [y1, y2]
	
model = my_model()
#model.load_state_dict(torch.load("output/resnet.pkl"))
model.load_state_dict(torch.load(model_file))

# loss and optimize
optimizer = optim.Adam(model.parameters(), lr=init_lr, weight_decay=0.0001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)
train_loss = []
train_loss_sum = 0.0
train_samples = 0
train_corrects = 0.0
train_corrects_angle = [0.0, 0.0, 0.0, 0.0, 0.0]
angle_tole = [2.37, 5.28, 8.53, 11.84, 15.17] 
		
def training_m0(save_loss=False):
	global train_loss_sum
	global train_samples
	global train_corrects 
	model.train()
	begin_time = time.time()

	for i, sample_real in enumerate(dataloaders['train']):
		xdata_real = Variable(sample_real['xdata'].cuda())
		ydata_real = [Variable(sample_real['ydata_bin'].cuda()), Variable(sample_real['ydata_res'].cuda())]
		#print(ydata_real)
		output_real = model(xdata_real)
		_, preds = torch.max(output_real[0], 1)
		loss = criterion1(output_real, ydata_real)
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
		
		#store	
		train_loss_sum += loss.data[0]
		train_corrects += torch.sum(preds == ydata_real[0].squeeze())
		train_samples += (xdata_real.size(0))
		if (i+1) % 10 == 0 and save_loss:
			batch_loss = train_loss_sum * batch_size['train'] / train_samples
			batch_acc = float(train_corrects.item()) / train_samples 
			#pdb.set_trace()
			print('train Batch [{}] Loss: {:.4f} Acc: {:.4f} Time: {:.4f}s'. 
		 format(i+1, batch_loss, batch_acc, time.time()-begin_time))
			begin_time = time.time()
			train_loss_sum = 0
			train_corrects = 0
			train_samples = 0
		if i>0 and i % 500 == 0:
			save_checkpoint("output/resnet_iter_" + str(i) + ".pkl")
	save_checkpoint("output/resnet.pkl")	

def training_m1(save_loss=False):
	global train_loss_sum
	global train_samples
	global train_corrects
	global train_corrects_angle
	model.train()
	begin_time = time.time()

	for i, sample_real in enumerate(dataloaders['train']):
		xdata_real = Variable(sample_real['xdata'].cuda())
		ydata_real = [Variable(sample_real['ydata_bin'].cuda()), Variable(sample_real['ydata_res'].cuda())]
		#print(ydata_real)
		output_real = model(xdata_real)
		_, preds = torch.max(output_real[0], 1)
		loss = criterion2(output_real, ydata_real)
		
		ypred, ytrue = output_real, ydata_real
		_, ind = torch.max(ypred[0], dim=1)
		torch_cluster_centers_ = Variable(torch.from_numpy(kmeans.cluster_centers_).float()).cuda()
		y1 = torch.index_select(torch_cluster_centers_, 0, ind)
		y2 = torch.index_select(torch_cluster_centers_, 0, ytrue[0].squeeze())
		ypred = (y1+ypred[1])/180*math.pi
		ytrue = (y2+ytrue[1].squeeze())/180*math.pi
		
		tmp = torch.sin(ypred[:,1])*torch.sin(ytrue[:,1]) + torch.cos(ypred[:,1])*torch.cos(ytrue[:,1])*torch.cos(ypred[:,0]-ytrue[:,0])
		dist = torch.acos(torch.clamp(tmp, -1+eps, 1-eps))*180/math.pi
		#pdb.set_trace()
					
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
		
		#store	
		train_loss_sum += loss.data[0]
		train_corrects += torch.sum(preds == ydata_real[0].squeeze())
		train_samples += (xdata_real.size(0))
		for j in range(len(angle_tole)):
			train_corrects_angle[j] += torch.sum(dist < angle_tole[j])
			
		batch_acc_angle = [0.0, 0.0, 0.0, 0.0, 0.0]
		if (i+1) % 10 == 0 and save_loss:
			batch_loss = train_loss_sum * batch_size['train'] / train_samples
			batch_acc = float(train_corrects.item()) / train_samples 
			#pdb.set_trace()
			for j in range(len(angle_tole)):
				batch_acc_angle[j] = float(train_corrects_angle[j].item()) / train_samples 
			
			print('train Batch [{}] Loss: {:.4f} Acc: {:.4f} Acc0: {:.4f} Acc1: {:.4f} Acc2: {:.4f} Acc3: {:.4f} Acc4: {:.4f} Time: {:.4f}s'. 
		 format(i+1, batch_loss, batch_acc, batch_acc_angle[0], batch_acc_angle[1], batch_acc_angle[2], batch_acc_angle[3], batch_acc_angle[4], time.time()-begin_time))
			begin_time = time.time()
			train_loss_sum = 0
			train_corrects = 0
			train_samples = 0
			train_corrects_angle = [0.0, 0.0, 0.0, 0.0, 0.0]
		if i>0 and i % 500 == 0:
			save_checkpoint("output/resnet_iter_" + str(i) + ".pkl")
	save_checkpoint("output/resnet.pkl")	

def testing():
	test_samples = 0
	test_corrects = 0.0
	test_corrects_angle = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
	acc_final = 0.0
	acc_angle_final = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
	count_batch = 0
	begin_time = time.time()
	model.eval()
	
	for i,sample in enumerate(dataloaders['val']):
		xdata = Variable(sample['xdata'].cuda())
		ydata = [Variable(sample['ydata_bin'].cuda()), Variable(sample['ydata_res'].cuda())]
		
		output = model(xdata)
		_, preds = torch.max(output[0], 1)
		
		ypred, ytrue = output, ydata
		_, ind = torch.max(ypred[0], dim=1)
		torch_cluster_centers_ = Variable(torch.from_numpy(kmeans.cluster_centers_).float()).cuda()
		y1 = torch.index_select(torch_cluster_centers_, 0, ind)
		y2 = torch.index_select(torch_cluster_centers_, 0, ytrue[0].squeeze())
		ypred = (y1+ypred[1])/180*math.pi
		ytrue = (y2+ytrue[1].squeeze())/180*math.pi
		
		tmp = torch.sin(ypred[:,1])*torch.sin(ytrue[:,1]) + torch.cos(ypred[:,1])*torch.cos(ytrue[:,1])*torch.cos(ypred[:,0]-ytrue[:,0])
		dist = torch.acos(torch.clamp(tmp, -1+eps, 1-eps))*180/math.pi
		#pdb.set_trace()
		
		#store	
		test_corrects += torch.sum(preds == ydata[0].squeeze())
		test_samples += (xdata.size(0))
		for j in range(len(angle_tole)):
			test_corrects_angle[j] += torch.sum(dist < angle_tole[j])
		
		batch_acc_angle = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
		if (i+1) % 10 == 0 :
			batch_acc = float(test_corrects.item()) / test_samples 
			#pdb.set_trace()
			for j in range(len(angle_tole)):
				batch_acc_angle[j] = float(test_corrects_angle[j].item()) / test_samples 
			
			print('test Batch [{}] Acc: {:.4f} Acc0: {:.4f} Acc1: {:.4f} Acc2: {:.4f} Acc3: {:.4f} Acc4: {:.4f} Time: {:.4f}s'. 
		 format(i+1, batch_acc, batch_acc_angle[0], batch_acc_angle[1], batch_acc_angle[2], batch_acc_angle[3], batch_acc_angle[4], time.time()-begin_time))
			begin_time = time.time()
			acc_final += batch_acc
			test_corrects = 0
			test_samples = 0
			acc_angle_final += batch_acc_angle 
			test_corrects_angle = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
			count_batch = count_batch+1
	
	acc_final /= count_batch
	acc_angle_final /= count_batch
	print('test final Acc: {:.4f} Acc0: {:.4f} Acc1: {:.4f} Acc2: {:.4f} Acc3: {:.4f}, Acc4: {:.4f}'. format(acc_final, acc_angle_final[0], acc_angle_final[1], acc_angle_final[2], acc_angle_final[3], acc_angle_final[4]))
		
		
		
def save_checkpoint(filename):
	torch.save(model.state_dict(), filename)
	
#training_m0(True)
testing()		

#trian_loss_sum = 0.0
#train_samples = 0
#for epoch in range(num_epochs):
#	tic = time.time()
#	scheduler.step()
#	# training step
#	training_m1(True)
#	# save model at end of epoch
#	save_checkpoint(model_file)
#	# time and output
#	toc = time.time() - tic
#	print('Epoch: {0} done in time {1}s'.format(epoch, toc))
#	testing()
	




	
	
	
