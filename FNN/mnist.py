from __future__ import print_function

import sys
import os
import time
import ipdb
import numpy as np
# np.random.seed(1234)  
from argparse import ArgumentParser

import theano
import theano.tensor as T

import lasagne

import cPickle as pickle
import gzip

import batch_norm
import laq
import optimizer

from pylearn2.datasets.mnist import MNIST
from pylearn2.utils import serial

from collections import OrderedDict


def main(method,LR_start):
	
	# BN parameters
	name = "mnist"
	print("dataset = "+str(name))
	print("Method = "+str(method))
	# alpha is the exponential moving average factor
	alpha = .1
	print("alpha = "+str(alpha))
	epsilon = 1e-4
	print("epsilon = "+str(epsilon))
	
	batch_size = 100
	print("batch_size = "+str(batch_size))

	num_epochs = 50
	print("num_epochs = "+str(num_epochs))

	# network structure
	num_units = 2048
	print("num_units = "+str(num_units))
	n_hidden_layers = 3
	print("n_hidden_layers = "+str(n_hidden_layers))

	print("LR_start = "+str(LR_start))
	LR_decay = 0.1
	print("LR_decay="+str(LR_decay))
	
	activation = lasagne.nonlinearities.rectify


	print('Loading MNIST dataset...')
	
	train_set = MNIST(which_set= 'train', start=0, stop = 50000, center = True)
	valid_set = MNIST(which_set= 'train', start=50000, stop = 60000, center = True)
	test_set = MNIST(which_set= 'test', center = True)
	
	# bc01 format
	train_set.X = train_set.X.reshape(-1, 1, 28, 28)
	valid_set.X = valid_set.X.reshape(-1, 1, 28, 28)
	test_set.X = test_set.X.reshape(-1, 1, 28, 28)
	
	# flatten targets
	train_set.y = np.hstack(train_set.y)
	valid_set.y = np.hstack(valid_set.y)
	test_set.y = np.hstack(test_set.y)
	
	# Onehot the targets
	train_set.y = np.float32(np.eye(10)[train_set.y])    
	valid_set.y = np.float32(np.eye(10)[valid_set.y])
	test_set.y = np.float32(np.eye(10)[test_set.y])
	
	# for hinge loss
	train_set.y = 2* train_set.y - 1.
	valid_set.y = 2* valid_set.y - 1.
	test_set.y = 2* test_set.y - 1.

	print('Building the MLP...') 
	
	# Prepare Theano variables for inputs and targets
	input = T.tensor4('inputs')
	target = T.matrix('targets')
	LR = T.scalar('LR', dtype=theano.config.floatX)

	mlp = lasagne.layers.InputLayer(
			shape=(None, 1, 28, 28),
			input_var=input)
	
	for k in range(n_hidden_layers):
		mlp = laq.DenseLayer(
				mlp, 
				nonlinearity=lasagne.nonlinearities.identity,
				num_units=num_units,
				method = method)                  	
		mlp = batch_norm.BatchNormLayer(
				mlp,
				epsilon=epsilon, 
				alpha=alpha)
		mlp = lasagne.layers.NonlinearityLayer(
				mlp,
				nonlinearity = activation)

	mlp = laq.DenseLayer(
				mlp, 
				nonlinearity=lasagne.nonlinearities.identity,
				num_units=10,
				method = method)      
				  
	mlp = batch_norm.BatchNormLayer(
			mlp,
			epsilon=epsilon, 
			alpha=alpha)

	train_output = lasagne.layers.get_output(mlp, deterministic=False)
	# squared hinge loss
	loss = T.mean(T.sqr(T.maximum(0.,1.-target*train_output)))
	

	if method!="FPN":
		
		# W updates
		W = lasagne.layers.get_all_params(mlp, quantized=True)
		W_grads = laq.compute_grads(loss,mlp)
		updates = optimizer.adam(loss_or_grads=W_grads, params=W, learning_rate=LR)
		updates = laq.clipping_scaling(updates,mlp)
		
		# other parameters updates
		params = lasagne.layers.get_all_params(mlp, trainable=True, quantized=False)
		updates = OrderedDict(updates.items() + optimizer.adam(loss_or_grads=loss, params=params, 
			learning_rate=LR, epsilon = 1e-8).items())


		## update the ternary matrix
		ternary_weights = laq.get_quantized_weights(loss, mlp)
		updates2 = OrderedDict()
		idx = 0
		tt_tag = lasagne.layers.get_all_params(mlp, tt=True)	
		for tt_tag_temp in tt_tag:
			updates2[tt_tag_temp]= ternary_weights[idx]
			idx = idx+1
		updates = OrderedDict(updates.items() + updates2.items())

		## update 2nd momentum
		updates3 = OrderedDict()
		acc_tag = lasagne.layers.get_all_params(mlp, acc=True)	
		idx = 0
		beta2 = 0.999
		for acc_tag_temp in acc_tag:
			updates3[acc_tag_temp]= acc_tag_temp*beta2 + W_grads[idx]*W_grads[idx]*(1-beta2)
			idx = idx+1

		updates = OrderedDict(updates.items() + updates3.items())

	else:
		params = lasagne.layers.get_all_params(mlp, trainable=True)
		updates = optimizer.adam(loss_or_grads=loss, params=params, learning_rate=LR)

	test_output = lasagne.layers.get_output(mlp, deterministic=True)
		
	test_loss = T.mean(T.sqr(T.maximum(0.,1.-target*test_output)))
	test_err = T.mean(T.neq(T.argmax(test_output, axis=1), T.argmax(target, axis=1)),dtype=theano.config.floatX)
	

	train_fn = theano.function([input, target, LR], loss, updates=updates)

	val_fn = theano.function([input, target], [test_loss, test_err])

	print('Training...')
	
	

	X_train = train_set.X
	y_train = train_set.y
	X_val = valid_set.X
	y_val = valid_set.y
	X_test = test_set.X
	y_test = test_set.y
	# This function trains the model a full epoch (on the whole dataset)
	def train_epoch(X,y,LR):
		
		loss = 0
		batches = len(X)/batch_size
		shuffled_range = range(len(X))
		np.random.shuffle(shuffled_range)
		for i in range(batches):
			tmp_ind = shuffled_range[i*batch_size:(i+1)*batch_size]  
			newloss = train_fn(X[tmp_ind],y[tmp_ind],LR) 
			loss +=newloss	

		loss/=batches		
		return loss
	
	# This function tests the model a full epoch (on the whole dataset)
	def val_epoch(X,y):
		
		err = 0
		loss = 0
		batches = len(X)/batch_size
		
		for i in range(batches):
			new_loss, new_err = val_fn(X[i*batch_size:(i+1)*batch_size], y[i*batch_size:(i+1)*batch_size])
			err += new_err
			loss += new_loss
		
		err = err / batches * 100
		loss /= batches

		return err, loss
	

	best_val_err = 100
	best_epoch = 1
	LR = LR_start
	# We iterate over epochs:
	for epoch in range(1, num_epochs+1):
		start_time = time.time()
		train_loss = train_epoch(X_train,y_train,LR)
		val_err, val_loss = val_epoch(X_val,y_val)

		# test if validation error went down
		if val_err <= best_val_err:
			best_val_err = val_err
			best_epoch = epoch
			test_err, test_loss = val_epoch(X_test,y_test)
			all_params = lasagne.layers.get_all_params(mlp)
			np.savez('{0}/{1}_lr{2}_{3}.npz'.format(method, name,  LR_start, method), *all_params)

		epoch_duration = time.time() - start_time
		
		# Then we print the results for this epoch:
		print("Epoch "+str(epoch)+" of "+str(num_epochs)+" took "+str(epoch_duration)+"s")
		print("  LR:                            "+str(LR))
		print("  training loss:                 "+str(train_loss))
		print("  validation loss:               "+str(val_loss))
		print("  validation error rate:         "+str(val_err)+"%")
		print("  best epoch:                    "+str(best_epoch))
		print("  best validation error rate:    "+str(best_val_err)+"%")
		print("  test loss:                     "+str(test_loss))
		print("  test error rate:               "+str(test_err)+"%") 
		

		with open("{0}/{1}_lr{2}_{3}.txt".format(method,name,  LR_start, method), "a") as myfile:
			myfile.write("{0}  {1:.5f} {2:.5f} {3:.5f} {4:.5f} {5:.5f} {6:.5f} {7:.5f}\n".format(epoch, 
				train_loss, val_loss, test_loss, val_err, test_err, epoch_duration, LR))

		# Learning rate update scheme
		if epoch == 15 or epoch==25:
			LR*=LR_decay


if __name__ == "__main__":
	parser = ArgumentParser()
	parser.add_argument("--method", type=str, dest="method",
				default="LATa", help="Method used, LAT-a, LAT_e, FPN")
	parser.add_argument("--lr_start",  type=float, dest="LR_start",
				default=0.01, help="Learning rate") 
	args = parser.parse_args()

	main(**vars(args))