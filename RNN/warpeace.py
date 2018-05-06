from __future__ import print_function

import sys
import os
import time

import numpy as np
np.random.seed(1234)  


import theano
import theano.tensor as T

import lasagne

import cPickle as pickle
import gzip

import laq
import optimizer

# from pylearn2.datasets.mnist import MNIST
# from pylearn2.utils import serial

from collections import OrderedDict
import ipdb
from argparse import ArgumentParser


def main(method,LR_start, SEQ_LENGTH):

	lasagne.random.set_rng(np.random.RandomState(1))

	name = "warpeace"
	print("dataset = "+str(name))

	print("Method = "+str(method))

	SEQ_LENGTH = SEQ_LENGTH
	# Sequence Length
	# SEQ_LENGTH = 50  #can have diffvalues 50, 100, 200
	print("SEQ_LENGTH = "+str(SEQ_LENGTH))

	# Number of units in the two hidden (LSTM) layers
	N_HIDDEN = 512
	print("N_HIDDEN = "+str(N_HIDDEN))

	# All gradients above this will be clipped
	GRAD_CLIP=5.  #### this clip the gradients at every time step, while T.clip clip the sum of gradients as a whole
	print("GRAD_CLIP ="+str(GRAD_CLIP))

	# Number of epochs to train the net
	num_epochs = 200
	print("num_epochs = "+str(num_epochs))

	# Batch Size
	batch_size = 100
	print("batch_size = "+str(batch_size))
	 
	print("LR_start = "+str(LR_start))
	LR_decay = 0.98
	print("LR_decay="+str(LR_decay))

	activation = lasagne.nonlinearities.tanh

	## load data, change the data file dir 
	with open('data/warpeace_input.txt', 'r') as f:
		in_text = f.read()
	generation_phrase = "With\r\nthese words she greeted Prince Vasili Kuragin, a man of high rank and\r\nimportance, who was the first to arrive at her reception. Anna Pavlovna\r\nhad had a cough for some days. She was, as sh"
	#This snippet loads the text file and creates dictionaries to 
	#encode characters into a vector-space representation and vice-versa. 
	chars = list(set(in_text))
	data_size, vocab_size = len(in_text), len(chars)
	char_to_ix = { ch:i for i,ch in enumerate(chars) }
	ix_to_char = { i:ch for i,ch in enumerate(chars) }

	num_splits = [0.8, 0.1, 0.1]
	num_splits_all = np.floor(data_size/batch_size/SEQ_LENGTH)
	num_train = np.floor(num_splits_all*num_splits[0])
	num_val   = np.floor(num_splits_all*num_splits[1])
	num_test  = num_splits_all - num_train - num_val

	train_X = in_text[0:(num_train*batch_size*SEQ_LENGTH+1).astype('int32')]
	val_X = in_text[(num_train*batch_size*SEQ_LENGTH).astype('int32'):((num_train+num_val)*batch_size*SEQ_LENGTH+1).astype('int32')]
	test_X = in_text[((num_train+num_val)*batch_size*SEQ_LENGTH).astype('int32'):(num_splits_all*batch_size*SEQ_LENGTH+1).astype('int32')]

	## build model
	print('Building the model...') 
	# input = T.tensor3('inputs')
	target = T.imatrix('target')
	LR = T.scalar('LR', dtype=theano.config.floatX)

	# (batch size, SEQ_LENGTH, num_features)
	l_in = lasagne.layers.InputLayer(shape=(None, None, vocab_size))
	l_forward_2 = laq.LSTMLayer(
				l_in, 
				num_units=N_HIDDEN,
				grad_clipping=GRAD_CLIP,
				peepholes=False,
				nonlinearity=activation, 
				method=method)   ### batch_size*SEQ_LENGTH*N_HIDDEN

	l_shp = lasagne.layers.ReshapeLayer(l_forward_2, (-1, N_HIDDEN))  ## (batch_size*SEQ_LENGTH, N_HIDDEN)
	l_out = lasagne.layers.DenseLayer(l_shp, num_units=vocab_size, W = lasagne.init.Normal(), nonlinearity=lasagne.nonlinearities.softmax)
	batchsize, seqlen, _ = l_in.input_var.shape

	train_output = lasagne.layers.get_output(l_out, deterministic=False)
	loss = T.nnet.categorical_crossentropy(train_output,target.flatten()).mean()



	if method!= "FPN": 
		# W updates
		W = lasagne.layers.get_all_params(l_out, quantized=True)
		W_grads = laq.compute_grads(loss,l_out) 
		updates = optimizer.adam(loss_or_grads=W_grads, params=W, learning_rate=LR, epsilon=1e-8) 
		updates = laq.clipping_scaling(updates,l_out)

		# other parameters updates
		params = lasagne.layers.get_all_params(l_out, trainable=True, quantized=False)
		updates = OrderedDict(updates.items() + optimizer.adam(loss_or_grads=loss, params=params, learning_rate=LR, epsilon = 1e-8).items())


		## update the ternary matrix
		ternary_weights = laq.get_quantized_weights(loss, l_out)
		updates2 = OrderedDict()
		idx = 0
		tt_tag = lasagne.layers.get_all_params(l_out, tt=True)	
		for tt_tag_temp in tt_tag:
			updates2[tt_tag_temp]= ternary_weights[idx]
			idx = idx+1
		updates = OrderedDict(updates.items() + updates2.items())

		## update 2nd momentum
		updates3 = OrderedDict()
		acc_tag = lasagne.layers.get_all_params(l_out, acc=True)	
		idx = 0
		beta2 = 0.999
		for acc_tag_temp in acc_tag:
			updates3[acc_tag_temp]= acc_tag_temp*beta2 + W_grads[idx]*W_grads[idx]*(1-beta2)
			idx = idx+1

		updates = OrderedDict(updates.items() + updates3.items())



	else:
		params_other = lasagne.layers.get_all_params(l_out, trainable=True)
		
		W_grads = [theano.grad(loss, wrt=l_forward_2.W_in_to_ingate), theano.grad(loss, wrt=l_forward_2.W_hid_to_ingate),
		theano.grad(loss, wrt=l_forward_2.W_in_to_forgetgate),theano.grad(loss, wrt=l_forward_2.W_hid_to_forgetgate),
		theano.grad(loss, wrt=l_forward_2.W_in_to_cell),theano.grad(loss, wrt=l_forward_2.W_hid_to_cell),
		theano.grad(loss, wrt=l_forward_2.W_in_to_outgate),theano.grad(loss, wrt=l_forward_2.W_hid_to_outgate)]
		
		updates = optimizer.adam(loss_or_grads=loss, params=params_other, learning_rate=LR)

	test_output = lasagne.layers.get_output(l_out, deterministic=True)
	test_loss = T.nnet.categorical_crossentropy(test_output,target.flatten()).mean()	
	train_fn = theano.function([l_in.input_var, target, LR], loss, updates=updates, allow_input_downcast=True)
	val_fn = theano.function([l_in.input_var, target], test_loss, allow_input_downcast=True)

	
	print('Training...')
	

	

	X_train = train_X
	X_val = val_X
	X_test = test_X

	def gen_data(pp, batch_size,SEQ_LENGTH, data, return_target=True):

		x = np.zeros((batch_size,SEQ_LENGTH,vocab_size))   ###### 128*100*85
		y = np.zeros((batch_size, SEQ_LENGTH))

		for n in range(batch_size):
			# ptr = n
			for i in range(SEQ_LENGTH):
				x[n,i,char_to_ix[data[pp[n]*SEQ_LENGTH+i]]] = 1.
				y[n,i] = char_to_ix[data[pp[n]*SEQ_LENGTH+i+1]]
		return x, np.array(y,dtype='int32')    

	in_text = X_train+X_val+X_test
	chars = list(set(in_text))
	data_size, vocab_size = len(in_text), len(chars)
	char_to_ix = { ch:i for i,ch in enumerate(chars) }
	ix_to_char = { i:ch for i,ch in enumerate(chars) }
	
	def train_epoch(X,LR):
		
		loss = 0        
		batches = len(X)/batch_size/SEQ_LENGTH
		num_seq = len(X)/SEQ_LENGTH
		shuffled_ind = range(num_seq)

		np.random.shuffle(shuffled_ind)
		
		for i in range(batches):
			tmp_ind = shuffled_ind[i*batch_size:(i+1)*batch_size]
			xx,yy = gen_data(tmp_ind,batch_size,SEQ_LENGTH, X)
			new_loss = train_fn(xx,yy,LR)
			loss+=new_loss

		loss=loss/batches
		
		return loss
	
	# This function tests the model a full epoch (on the whole dataset)
	def val_epoch(X):
		
		# err = 0
		loss = 0
		batches = len(X)/batch_size/SEQ_LENGTH

		num_seq = len(X)/SEQ_LENGTH
		ind = range(num_seq)
		for i in range(batches):
			tmp_ind = ind[i*batch_size:(i+1)*batch_size]
			xx, yy = gen_data(tmp_ind, batch_size, SEQ_LENGTH, X)
			new_loss = val_fn(xx,yy)
			loss += new_loss
		
		loss = loss/batches

		return loss
	
	best_val_loss=100
	best_epoch = 1
	LR = LR_start

	# iterate over epochs:
	for epoch in range(1,num_epochs+1):		
		start_time = time.time()
		train_loss = train_epoch(X_train, LR)

		val_loss = val_epoch(X_val)
		
		# test if validation error went down
		if val_loss <= best_val_loss:
			
			best_val_loss = val_loss
			best_epoch = epoch
			
			test_loss = val_epoch(X_test)

			# all_params = lasagne.layers.get_all_params(l_out)
			# np.savez("{0}/{1}_seq{2}_lr{3}_hid{4}_{5}.npz".format(method, name, SEQ_LENGTH, LR_start, N_HIDDEN, method), *all_params)		

		epoch_duration = time.time() - start_time
		# Then we print the results for this epoch:
		print("  Epoch "+str(epoch)+" of "+str(num_epochs)+" took "+str(epoch_duration)+"s")
		print("  LR:                            "+str(LR))
		print("  training loss:                 "+str(train_loss))
		print("  validation loss:               "+str(val_loss))
		print("  best epoch:                    "+str(best_epoch))
		print("  test loss:                     "+str(test_loss))
		
		with open("{0}/{1}_seq{2}_lr{3}_hid{4}_{5}.txt".format(method, name, SEQ_LENGTH, LR_start, N_HIDDEN, method), "a") as myfile:
			myfile.write("{0}  {1:.3f} {2:.3f} {3:.3f} {4:.3f}\n".format(epoch, train_loss, val_loss, 
				test_loss, epoch_duration))

		# learning rate update scheme
		if epoch>10:
			LR *= LR_decay


if __name__ == "__main__":
	parser = ArgumentParser()
	parser.add_argument("--method", type=str, dest="method",
				default="LATa", help="Method used, combination, exact, heuristic,, FPN")
	parser.add_argument("--lr_start",  type=float, dest="LR_start",
				default=2e-3, help="Learning rate") 
	parser.add_argument("--len", type=int, dest="SEQ_LENGTH",
				default=100, help="unrolled timesteps for LSTM")
	args = parser.parse_args()

	main(**vars(args))
