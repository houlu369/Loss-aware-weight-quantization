import time

from collections import OrderedDict

import numpy as np
np.random.seed(1234)  
import ipdb

import theano
import theano.tensor as T

import lasagne
from theano.scalar.basic import UnaryScalarOp, same_out_nocomplex
from theano.tensor.elemwise import Elemwise
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
class Round3(UnaryScalarOp):
	
	def c_code(self, node, name, (x,), (z,), sub):
		return "%(z)s = round(%(x)s);" % locals()
	
	def grad(self, inputs, gout):
		(gz,) = gout
		return gz, 
		
round3_scalar = Round3(same_out_nocomplex, name='round3')
round3 = Elemwise(round3_scalar)

def hard_sigmoid(x):
	return T.clip((x+1.)/2.,0,1)

def binary_tanh_unit(x):
	return 2.*round3(hard_sigmoid(x))-1.
	
def binary_sigmoid_unit(x):
	return round3(hard_sigmoid(x))


def findalpha(D, W):
	W = T.abs_(T.flatten(W)) 
	D = T.flatten(D)
#     sorted_W = T.sort(W)[::-1]
	ind = T.argsort(W)[::-1]
	cum_DW = T.cumsum(T.abs_(D*W)[ind])
	cum_D = T.cumsum(D[ind])
	cum_DW_D = cum_DW/cum_D/2	# tmp = W[ind] - cum_DW_D
	tmp = W[ind][:-1] - cum_DW_D[:-1]
	tmp1 = W[ind][1:] - cum_DW_D[:-1]
	tmp3 = tmp*tmp1
	
	mask = T.lt(tmp3, 0)

	tmp4 = mask.nonzero()[0].shape[0]   
	tmp5 = cum_DW_D[mask.nonzero()]*cum_DW_D[mask.nonzero()]*cum_D[mask.nonzero()]
	bb = cum_DW_D[mask.nonzero()][T.argmax(tmp5)]	

	from theano.ifelse import ifelse
	thres =  ifelse(T.gt(tmp4,0), bb, 0.7*cum_DW_D[-1] ) 

	return thres

def findalpha2(D, W):
	W = T.flatten(W)
	D = T.flatten(D)
	# the positive  part 
	n1 = T.sum(T.gt(W,0.))
	ind1 = T.argsort(W)[::-1]
	cum_DW1 = T.cumsum(T.abs_(D*W)[ind1])
	cum_D1 = T.cumsum(D[ind1])
	c1 = cum_DW1/cum_D1/2	# tmp = W[ind] - cum_DW_D
	mask1 = T.lt((W[ind1][0:n1-1] - c1[0:n1-1])*(W[ind1][1:n1] - c1[0:n1-1]), 0)
	thr1 = c1[mask1.nonzero()][T.argmax(c1[mask1.nonzero()]*c1[mask1.nonzero()]*cum_D1[mask1.nonzero()])]	
	from theano.ifelse import ifelse
	thres1 =  ifelse(T.gt(mask1.nonzero()[0].shape[0],0), thr1, 0.7*c1[n1-1] ) 
	# the negative part 
	n2 = T.sum(T.lt(W,0.))
	ind2 = ind1[::-1]
	cum_DW2 = T.cumsum(T.abs_(D*W)[ind2])
	cum_D2 = T.cumsum(D[ind2])
	c2 = cum_DW2/cum_D2/2	# tmp = W[ind] - cum_DW_D
	mask2 = T.lt((-W[ind2][0:n2-1] - c2[0:n2-1])*(-W[ind2][1:n2] - c2[0:n2-1]), 0)
	thr2 = c2[mask2.nonzero()][T.argmax(c1[mask2.nonzero()]*c1[mask2.nonzero()]*cum_D2[mask2.nonzero()])]	
	from theano.ifelse import ifelse
	thres2 =  ifelse(T.gt(mask2.nonzero()[0].shape[0],0), thr2, 0.7*c2[n2-1] ) 
	return thres1, thres2

# The quantization function
def quantization(W,Wacc,method, Wb):
	
	if method == "FPN":
		Wb = W

	elif method == "LAB":
		L = (T.sqrt(Wacc) + 1e-8) 
		Wb = hard_sigmoid(W)
		Wb = round3(Wb)
		Wb = T.cast(T.switch(Wb,1.,-1.), theano.config.floatX) 

		alpha  = (T.abs_(L*W).sum()/L.sum()).astype('float32') 
		Wb = alpha*Wb		
		
	elif method=="LATa":
		D = (T.sqrt(Wacc) + 1e-8) 
		b = T.sgn(Wb)

		alpha  = (T.abs_(b*D*W).sum()/T.abs_(b*D).sum()).astype('float32') 
		b = T.switch(T.gt(W/alpha, 0.5), 1., T.switch(T.lt(W/alpha, -0.5), -1., 0.) )
		def OneStep(alpha, b):
			# minimize alpha
			alpha_new  = (T.abs_(b*D*W).sum()/T.abs_(b*D).sum()).astype('float32') 
			# minimize b
			b_new = T.switch(T.gt(W/alpha_new, 0.5), 1., T.switch(T.lt(W/alpha_new, -0.5), -1., 0.))
			delta = T.abs_(alpha_new-alpha)
			condition = T.lt(delta, 1e-6)
			return [alpha_new, b_new], theano.scan_module.until(condition)

		[out1, out2], updates = theano.scan(fn=OneStep ,outputs_info=[alpha, b],n_steps=10) 
		Wb  = out1[-1]*out2[-1]

	elif method=="LATe":
		D = (T.sqrt(Wacc) + 1e-8) 
		thres = findalpha(D, W)
		alpha = thres*2
		Wt = T.switch(T.gt(W, thres), 1., T.switch(T.lt(W, -thres), -1., 0.) )
		Wb = alpha*Wt

	elif method=="LAT2e":
		D = (T.sqrt(Wacc) + 1e-8) 
		thres1, thres2 = findalpha2(D, W)
		alpha1 = thres1*2
		Wt1 = T.switch(T.gt(W, thres1), 1., 0.) 
		alpha2 = thres2*2
		Wt2 = T.switch(T.lt(W, -thres2), -1., 0.) 

		Wb = alpha1*Wt1 + alpha2*Wt2	

	elif method=="LAT2a":
		D = (T.sqrt(Wacc) + 1e-8) 
		b1 = T.ge(Wb,0)
		alpha1 = (T.abs_(b1*D*W).sum()/T.abs_(b1*D).sum()).astype('float32') 
		b1 = T.switch(T.gt(W/alpha1, 0.5), 1., 0.)
		b2 =  T.lt(Wb,0)
		alpha2 = (T.abs_(b2*D*W).sum()/T.abs_(b2*D).sum()).astype('float32') 
		b2 = T.switch(T.lt(W/alpha2, -0.5), -1., 0.)

		def OneStep(alpha1, b1, alpha2, b2):
				alpha1_new  = (T.abs_(b1*D*W).sum()/T.abs_(b1*D).sum()).astype('float32') 
				b1_new = T.switch(T.gt(W/alpha1_new, 0.5), 1., 0.)
				alpha2_new = (T.abs_(b2*D*W).sum()/T.abs_(b2*D).sum()).astype('float32') 
				b2_new = T.switch(T.lt(W/alpha2_new, -0.5), -1., 0.)

				delta1 = T.abs_(alpha1_new-alpha1)
				delta2 = T.abs_(alpha2_new-alpha2)
				condition = T.lt(delta1, 1e-6) and T.lt(delta2, 1e-6)
				return [alpha1_new, b1_new, alpha2_new, b2_new], theano.scan_module.until(condition)

		[out1, out2, out3, out4], updates = theano.scan(fn=OneStep ,outputs_info=[alpha1, b1, alpha2, b2],n_steps=10)
		Wb  = out1[-1]*out2[-1] + out3[-1]*out4[-1]	
								  

	elif method=="LAQ_linear":
		D = (T.sqrt(Wacc) + 1e-8) 
		b = T.sgn(Wb)
		alpha  = (T.abs_(b*D*W).sum()/T.abs_(b*D).sum()).astype('float32') 
		m = 3  ## k-bit 
		n = 2**(m-1)-1

		b = round3(T.clip(W/alpha, -1., 1.)*n)/(n)
		def OneStep(alpha, b):
				# minimize alpha
				alpha_new  = (T.abs_(b*D*W).sum()/T.abs_(b*D).sum()).astype('float32') 
				# minimize b
				b_new = round3(T.clip(W/alpha_new, -1., 1.)*n)/(n)
				delta = T.abs_(alpha_new-alpha)
				condition = T.lt(delta, 1e-6)
				return [alpha_new, b_new], theano.scan_module.until(condition)

		[out1, out2], updates = theano.scan(fn=OneStep ,outputs_info=[alpha, b],n_steps=10)
		Wb  = out1[-1]*out2[-1]		

	elif method=="LAQ_log":
		D = (T.sqrt(Wacc) + 1e-8) 
		b = T.sgn(Wb)
		alpha  = (T.abs_(b*D*W).sum()/T.abs_(b*D).sum()).astype('float32') 
		m = 3  ## m-bit 0, 1/4, 1/2, 1
		n = 2**(m-1)-1 # n here is the k in the paper
		tmp = T.clip(W/alpha, -1., 1.)
		# log2(1/2*(2^(-n)+2^(-(n+1)))) - (-n-(n+1))/2 = 0.0849625
		b =  T.switch( T.ge(tmp, pow(2, -n)), T.pow(2, round3(T.log2(tmp)-0.0849625)), 
			T.switch( T.le(tmp, -pow(2,-n)), -T.pow(2, round3(T.log2(-tmp)-0.0849625)), 0.))
		b = T.switch(T.ge(b, pow(2, - (n-1))), b, T.switch(T.le(b, -pow(2, -(n-1))), b, T.sgn(b)*pow(2,-(n-1))))
		def OneStep(alpha, b):
				# minimize alpha
				alpha_new  = (T.abs_(b*D*W).sum()/T.abs_(b*D).sum()).astype('float32') 
				# minimize b
				tmp_new = T.clip(W/alpha_new, -1., 1.)
				b_new =  T.switch( T.ge(tmp_new, pow(2, -n)), T.pow(2, round3(T.log2(tmp_new)-0.0849625)), 
					T.switch( T.le(tmp_new, -pow(2, -n)), -T.pow(2, round3(T.log2(-tmp_new)-0.0849625)), 0.))		
				b_new = T.switch(T.ge(b_new, pow(2, - (n-1))), b_new, 
					T.switch(T.le(b_new, -pow(2, -(n-1))), b_new, T.sgn(b_new)*pow(2, -(n-1))))
				delta = T.abs_(alpha_new-alpha)
				condition = T.lt(delta, 1e-6)
				return [alpha_new, b_new], theano.scan_module.until(condition)

		[out1, out2], updates = theano.scan(fn=OneStep ,outputs_info=[alpha, b],n_steps=10)
		Wb  = out1[-1]*out2[-1]	

	return Wb

def initialize_b(W): 
    thres = 0.7*np.sum(np.absolute(W))/W.size
    condlist = [W<-thres, W>thres]
    choicelist = [-thres*2, thres*2]
    b = np.select(condlist, choicelist).astype('float32')
    return b

# This class extends the Lasagne LSTM to support LAQ
class LSTMLayer(lasagne.layers.LSTMLayer):
	
	def __init__(self, incoming, num_units, method, **kwargs):
		
		self.method = method
		g_init = 0.08
		if self.method!="FPN":
			super(LSTMLayer, self).__init__(incoming, num_units, 
				ingate=lasagne.layers.Gate(W_in=lasagne.init.Uniform((-g_init, g_init)), 
					W_hid=lasagne.init.Uniform((-g_init, g_init)),
					W_cell=lasagne.init.Uniform((-g_init, g_init))),
				forgetgate=lasagne.layers.Gate(W_in=lasagne.init.Uniform((-g_init, g_init)), 
					W_hid=lasagne.init.Uniform((-g_init, g_init)),
					W_cell=lasagne.init.Uniform((-g_init, g_init)),
					b=lasagne.init.Constant(1.)), 
				cell=lasagne.layers.Gate(W_in=lasagne.init.Uniform((-g_init, g_init)), 
					W_hid=lasagne.init.Uniform((-g_init, g_init)),
					W_cell=None, nonlinearity=lasagne.nonlinearities.tanh),
				outgate=lasagne.layers.Gate(W_in=lasagne.init.Uniform((-g_init, g_init)), 
					W_hid=lasagne.init.Uniform((-g_init, g_init)),
					W_cell=lasagne.init.Uniform((-g_init, g_init))),
				**kwargs)
			# super(LSTMLayer, self).__init__(incoming, num_units, 
			#     **kwargs)
			# add the quantized tag to weights            
			self.params[self.W_in_to_ingate]=set(['quantized'])
			self.params[self.W_hid_to_ingate]=set(['quantized'])

			self.params[self.W_in_to_forgetgate]=set(['quantized'])
			self.params[self.W_hid_to_forgetgate]=set(['quantized'])

			self.params[self.W_in_to_cell]=set(['quantized'])
			self.params[self.W_hid_to_cell]=set(['quantized'])

			self.params[self.W_in_to_outgate]=set(['quantized'])
			self.params[self.W_hid_to_outgate]=set(['quantized'])

		else:
			# super(LSTMLayer, self).__init__(incoming, num_units, **kwargs)
			super(LSTMLayer, self).__init__(incoming, num_units, 
				ingate=lasagne.layers.Gate(W_in=lasagne.init.Uniform((-g_init, g_init)), 
					W_hid=lasagne.init.Uniform((-g_init, g_init)),
					W_cell=lasagne.init.Uniform((-g_init, g_init))),
				forgetgate=lasagne.layers.Gate(W_in=lasagne.init.Uniform((-g_init, g_init)), 
					W_hid=lasagne.init.Uniform((-g_init, g_init)),
					W_cell=lasagne.init.Uniform((-g_init, g_init)),
					b=lasagne.init.Constant(1.)), 
				cell=lasagne.layers.Gate(W_in=lasagne.init.Uniform((-g_init, g_init)), 
					W_hid=lasagne.init.Uniform((-g_init, g_init)),
					W_cell=None, nonlinearity=lasagne.nonlinearities.tanh),
				outgate=lasagne.layers.Gate(W_in=lasagne.init.Uniform((-g_init, g_init)), 
					W_hid=lasagne.init.Uniform((-g_init, g_init)),
					W_cell=lasagne.init.Uniform((-g_init, g_init))),
				**kwargs)


		# initialize 2nd moment
		self.acc_W_in_to_ingate = theano.shared(np.zeros((self.W_in_to_ingate.get_value(borrow=True)).shape, dtype='float32'))
		self.acc_W_hid_to_ingate = theano.shared(np.zeros((self.W_hid_to_ingate.get_value(borrow=True)).shape, dtype='float32'))
		self.acc_W_in_to_forgetgate = theano.shared(np.zeros((self.W_in_to_forgetgate.get_value(borrow=True)).shape, dtype='float32'))
		self.acc_W_hid_to_forgetgate = theano.shared(np.zeros((self.W_hid_to_forgetgate.get_value(borrow=True)).shape, dtype='float32'))
		self.acc_W_in_to_cell = theano.shared(np.zeros((self.W_in_to_cell.get_value(borrow=True)).shape, dtype='float32'))
		self.acc_W_hid_to_cell = theano.shared(np.zeros((self.W_hid_to_cell.get_value(borrow=True)).shape, dtype='float32'))
		self.acc_W_in_to_outgate = theano.shared(np.zeros((self.W_in_to_outgate.get_value(borrow=True)).shape, dtype='float32'))
		self.acc_W_hid_to_outgate = theano.shared(np.zeros((self.W_hid_to_outgate.get_value(borrow=True)).shape, dtype='float32'))
		
		self.params[self.acc_W_in_to_ingate]=set(['acc'])
		self.params[self.acc_W_hid_to_ingate]=set(['acc'])

		self.params[self.acc_W_in_to_forgetgate]=set(['acc'])
		self.params[self.acc_W_hid_to_forgetgate]=set(['acc'])

		self.params[self.acc_W_in_to_cell]=set(['acc'])
		self.params[self.acc_W_hid_to_cell]=set(['acc'])

		self.params[self.acc_W_in_to_outgate]=set(['acc'])
		self.params[self.acc_W_hid_to_outgate]=set(['acc'])


		# initialize quantized weights
		# self.bW_in_to_ingate = theano.shared(np.zeros((self.W_in_to_ingate.get_value(borrow=True)).shape, dtype='float32'))
		self.bW_in_to_ingate = theano.shared(initialize_b(self.W_in_to_ingate.get_value(borrow=True)))
		self.bW_hid_to_ingate = theano.shared(initialize_b(self.W_hid_to_ingate.get_value(borrow=True)))
		self.bW_in_to_forgetgate = theano.shared(initialize_b(self.W_in_to_forgetgate.get_value(borrow=True)))
		self.bW_hid_to_forgetgate = theano.shared(initialize_b(self.W_hid_to_forgetgate.get_value(borrow=True)))
		self.bW_in_to_cell = theano.shared(initialize_b(self.W_in_to_cell.get_value(borrow=True)))
		self.bW_hid_to_cell = theano.shared(initialize_b(self.W_hid_to_cell.get_value(borrow=True)))
		self.bW_in_to_outgate = theano.shared(initialize_b(self.W_in_to_outgate.get_value(borrow=True)))
		self.bW_hid_to_outgate = theano.shared(initialize_b(self.W_hid_to_outgate.get_value(borrow=True)))
		
		self.params[self.bW_in_to_ingate]=set(['tt'])
		self.params[self.bW_hid_to_ingate]=set(['tt'])

		self.params[self.bW_in_to_forgetgate]=set(['tt'])
		self.params[self.bW_hid_to_forgetgate]=set(['tt'])

		self.params[self.bW_in_to_cell]=set(['tt'])
		self.params[self.bW_hid_to_cell]=set(['tt'])

		self.params[self.bW_in_to_outgate]=set(['tt'])
		self.params[self.bW_hid_to_outgate]=set(['tt'])


	def get_output_for(self, input, deterministic=False, **kwargs):
		# self.Wb = quantization(self.W,self.H,self.quantized,deterministic,self.stochastic,self._srng)
		self.bW_in_to_ingate = quantization(self.W_in_to_ingate, self.acc_W_in_to_ingate, self.method, self.bW_in_to_ingate)
		self.bW_hid_to_ingate= quantization(self.W_hid_to_ingate, self.acc_W_hid_to_ingate, self.method, self.bW_hid_to_ingate)

		self.bW_in_to_forgetgate  = quantization(self.W_in_to_forgetgate, self.acc_W_in_to_forgetgate, self.method, self.bW_in_to_forgetgate)
		self.bW_hid_to_forgetgate   = quantization(self.W_hid_to_forgetgate, self.acc_W_hid_to_forgetgate, self.method, self.bW_hid_to_forgetgate)

		self.bW_in_to_cell = quantization(self.W_in_to_cell, self.acc_W_in_to_cell, self.method, self.bW_in_to_cell)
		self.bW_hid_to_cell = quantization(self.W_hid_to_cell, self.acc_W_hid_to_cell, self.method, self.bW_hid_to_cell)

		self.bW_in_to_outgate  = quantization(self.W_in_to_outgate, self.acc_W_in_to_outgate, self.method, self.bW_in_to_outgate)
		self.bW_hid_to_outgate   = quantization(self.W_hid_to_outgate, self.acc_W_hid_to_outgate, self.method, self.bW_hid_to_outgate)

		# Wr = self.W
		rW_in_to_ingate = self.W_in_to_ingate
		rW_hid_to_ingate = self.W_hid_to_ingate

		rW_in_to_forgetgate = self.W_in_to_forgetgate
		rW_hid_to_forgetgate = self.W_hid_to_forgetgate

		rW_in_to_cell = self.W_in_to_cell
		rW_hid_to_cell = self.W_hid_to_cell

		rW_in_to_outgate = self.W_in_to_outgate
		rW_hid_to_outgate = self.W_hid_to_outgate

		#self.W = bW
		self.W_in_to_ingate = self.bW_in_to_ingate
		self.W_hid_to_ingate = self.bW_hid_to_ingate

		self.W_in_to_forgetgate = self.bW_in_to_forgetgate
		self.W_hid_to_forgetgate = self.bW_hid_to_forgetgate

		self.W_in_to_cell = self.bW_in_to_cell
		self.W_hid_to_cell = self.bW_hid_to_cell

		self.W_in_to_outgate = self.bW_in_to_outgate
		self.W_hid_to_outgate = self.bW_hid_to_outgate
		
		rvalue = super(LSTMLayer, self).get_output_for(input, **kwargs)
		
		self.W_in_to_ingate = rW_in_to_ingate
		self.W_hid_to_ingate = rW_hid_to_ingate

		self.W_in_to_forgetgate = rW_in_to_forgetgate
		self.W_hid_to_forgetgate = rW_hid_to_forgetgate

		self.W_in_to_cell = rW_in_to_cell
		self.W_hid_to_cell = rW_hid_to_cell

		self.W_in_to_outgate = rW_in_to_outgate
		self.W_hid_to_outgate = rW_hid_to_outgate
		
		return rvalue



def compute_grads(loss,network):
		
	layers = lasagne.layers.get_all_layers(network)
	grads = []
	all_bW = []
	
	for layer in layers:
	
		params = layer.get_params(quantized=True)

		if params:
			all_bW = all_bW + [layer.bW_in_to_ingate, layer.bW_hid_to_ingate,layer.bW_in_to_forgetgate, 
			layer.bW_hid_to_forgetgate, layer.bW_in_to_cell,layer.bW_hid_to_cell,
			layer.bW_in_to_outgate, layer.bW_hid_to_outgate]
	
	grads = theano.grad(loss, all_bW)

	return grads

def get_quantized_weights(loss,network):
		
	layers = lasagne.layers.get_all_layers(network)
	quantized_weights = []
	
	for layer in layers:
	
		params = layer.get_params(quantized=True)
		if params:
			# print(params[0].name)
			quantized_weights.append(layer.bW_in_to_ingate)
			quantized_weights.append(layer.bW_hid_to_ingate)

			quantized_weights.append(layer.bW_in_to_forgetgate)
			quantized_weights.append(layer.bW_hid_to_forgetgate)

			quantized_weights.append(layer.bW_in_to_cell)
			quantized_weights.append(layer.bW_hid_to_cell)
			
			quantized_weights.append(layer.bW_in_to_outgate)
			quantized_weights.append(layer.bW_hid_to_outgate)

	return quantized_weights

def clipping_scaling(updates,network):
	
	layers = lasagne.layers.get_all_layers(network)
	updates = OrderedDict(updates)
	
	for layer in layers:	
		params = layer.get_params(quantized=True)
		for param in params:
			updates[param] = T.clip(updates[param], -1.,1.)     
	return updates
 

