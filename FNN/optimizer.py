
from collections import OrderedDict

import numpy as np

import theano
import theano.tensor as T
# from . import utils


def get_or_compute_grads(loss_or_grads, params):
    if any(not isinstance(p, theano.compile.SharedVariable) for p in params):
        raise ValueError("params must contain shared variables only. If it "
                         "contains arbitrary parameter expressions, then "
                         "lasagne.utils.collect_shared_vars() may help you.")
    if isinstance(loss_or_grads, list):
        if not len(loss_or_grads) == len(params):
            raise ValueError("Got %d gradient expressions for %d parameters" %
                             (len(loss_or_grads), len(params)))
        return loss_or_grads
    else:
        return theano.grad(loss_or_grads, params)


def sgd(loss_or_grads, params, learning_rate):
    grads = get_or_compute_grads(loss_or_grads, params)
    updates = OrderedDict()

    for param, grad in zip(params, grads):
        updates[param] = param - learning_rate * grad

    return updates


def apply_momentum(updates, params=None, momentum=0.9):
    if params is None:
        params = updates.keys()
    updates = OrderedDict(updates)

    for param in params:
        value = param.get_value(borrow=True)
        velocity = theano.shared(np.zeros(value.shape, dtype=value.dtype),
                                 broadcastable=param.broadcastable)
        x = momentum * velocity + updates[param]
        updates[velocity] = x - param
        updates[param] = x

    return updates


def momentum(loss_or_grads, params, learning_rate, momentum=0.9):
    updates = sgd(loss_or_grads, params, learning_rate)
    return apply_momentum(updates, momentum=momentum)


def apply_nesterov_momentum(updates, params=None, momentum=0.9):
    if params is None:
        params = updates.keys()
    updates = OrderedDict(updates)

    for param in params:
        value = param.get_value(borrow=True)
        velocity = theano.shared(np.zeros(value.shape, dtype=value.dtype),
                                 broadcastable=param.broadcastable)
        x = momentum * velocity + updates[param] - param
        updates[velocity] = x
        updates[param] = momentum * x + updates[param]

    return updates


def nesterov_momentum(loss_or_grads, params, learning_rate, momentum=0.9):
    updates = sgd(loss_or_grads, params, learning_rate)
    return apply_nesterov_momentum(updates, momentum=momentum)


def adagrad(loss_or_grads, params, learning_rate=1.0, epsilon=1e-6):
    grads = get_or_compute_grads(loss_or_grads, params)
    updates = OrderedDict()

    for param, grad in zip(params, grads):
        value = param.get_value(borrow=True)
        accu = theano.shared(np.zeros(value.shape, dtype=value.dtype),
                             broadcastable=param.broadcastable)
        accu_new = accu + grad ** 2
        updates[accu] = accu_new
        updates[param] = param - (learning_rate * grad /
                                  T.sqrt(accu_new + epsilon))

    return updates


def rmsprop(loss_or_grads, params, learning_rate=1.0, rho=0.9, epsilon=1e-6):
    grads = get_or_compute_grads(loss_or_grads, params)
    updates = OrderedDict()

    # Using theano constant to prevent upcasting of float32
    one = T.constant(1)

    for param, grad in zip(params, grads):
        value = param.get_value(borrow=True)
        accu = theano.shared(np.zeros(value.shape, dtype=value.dtype),
                             broadcastable=param.broadcastable)
        accu_new = rho * accu + (one - rho) * grad ** 2
        updates[accu] = accu_new
        updates[param] = param - (learning_rate * grad /
                                  T.sqrt(accu_new + epsilon))

    return updates


def adadelta(loss_or_grads, params, learning_rate=1.0, rho=0.95, epsilon=1e-6):
    grads = get_or_compute_grads(loss_or_grads, params)
    updates = OrderedDict()

    # Using theano constant to prevent upcasting of float32
    one = T.constant(1)

    for param, grad in zip(params, grads):
        value = param.get_value(borrow=True)
        # accu: accumulate gradient magnitudes
        accu = theano.shared(np.zeros(value.shape, dtype=value.dtype),
                             broadcastable=param.broadcastable)
        # delta_accu: accumulate update magnitudes (recursively!)
        delta_accu = theano.shared(np.zeros(value.shape, dtype=value.dtype),
                                   broadcastable=param.broadcastable)

        # update accu (as in rmsprop)
        accu_new = rho * accu + (one - rho) * grad ** 2
        updates[accu] = accu_new

        # compute parameter update, using the 'old' delta_accu
        update = (grad * T.sqrt(delta_accu + epsilon) /
                  T.sqrt(accu_new + epsilon))
        updates[param] = param - learning_rate * update

        # update delta_accu (as accu, but accumulating updates)
        delta_accu_new = rho * delta_accu + (one - rho) * update ** 2
        updates[delta_accu] = delta_accu_new

    return updates


def adam(loss_or_grads, params, learning_rate=0.001, beta1=0.9,
		 beta2=0.999, epsilon=1e-8):
	# if isinstance(loss_or_grads, list):
	# 	if not len(loss_or_grads) == len(params):
	# 		raise ValueError("Got %d gradient expressions for %d parameters" %
	# 						 (len(loss_or_grads), len(params)))
	# 	all_grads = loss_or_grads
	# else:
	# 	all_grads = theano.grad(loss_or_grads, params)
	all_grads = get_or_compute_grads(loss_or_grads, params)
	# t_prev = theano.shared(lasagne.utils.floatX(0.))
	t_prev = theano.shared(np.float32(0.))
	updates = OrderedDict()

	t = t_prev + 1
	a_t = learning_rate*T.sqrt(1-beta2**t)/(1-beta1**t)

	for param, g_t in zip(params, all_grads):
		value = param.get_value(borrow=True)
		m_prev = theano.shared(np.zeros(value.shape, dtype=value.dtype),
							   broadcastable=param.broadcastable)
		v_prev = theano.shared(np.zeros(value.shape, dtype=value.dtype),
							   broadcastable=param.broadcastable)

		m_t = beta1*m_prev + (1-beta1)*g_t
		v_t = beta2*v_prev + (1-beta2)*g_t**2
		step = a_t*m_t/(T.sqrt(v_t) + epsilon)

		updates[m_prev] = m_t
		updates[v_prev] = v_t
		updates[param] = param - step

	updates[t_prev] = t
	return updates