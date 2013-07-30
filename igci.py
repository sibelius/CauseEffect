import numpy as np
import math
from scipy.special import psi

def igci(x,y,refMeasure,estimator):
#function f = igci(x,y,refMeasure,estimator)
# Performs causal inference in a deterministic scenario (see [1] for details)
# Information Geometric Causal Inference (IGCI)
#
# USAGE:
#   f = igci(x,y,refMeasure,estimator)
# 
# INPUT:
#   x  - m x 1 observations of x
#   y  - m x 1 observations of y
#   refMeasure - reference measure to use:
#  1: uniform
#  2: Gaussian
#   estimator -  estimator to use:
#  1: entropy (eq. (12) in [1]),
#  2: integral approximation (eq. (13) in [1]).
# 
# OUTPUT: 
#   f > 0:   the method prefers the causal direction x -> y
#   f < 0:   the method prefers the causal direction y -> x
# 
# EXAMPLE: 
#   x = randn(100,1) y = exp(x) igci(x,y,2,1) > 0
#
#
# Copyright (c) 2010  Povilas Daniusis, Joris Mooij
# All rights reserved.  See the file COPYING for license terms.
# ----------------------------------------------------------------------------
#
# [1]  P. Daniusis, D. Janzing, J. Mooij, J. Zscheischler, B. Steudel,
#  K. Zhang, B. Scholkopf:  Inferring deterministic causal relations.
#  Proceedings of the 26th Annual Conference on Uncertainty in Artificial 
#  Intelligence (UAI-2010).  
#  http://event.cwi.nl/uai2010/papers/UAI2010_0121.pdf
#
# Cause-effect pair challenge modification
# Isabelle Guyon, February 2013:
# UI modified  to provide default parameter values and to
# change the sign of the output for compatibility reasons.

#if nargin<3, refMeasure=2 end
#if nargin<4, estimator=1 end

#if nargin <2 || nargin>4
#  help igci
#  error('Incorrect number of input arguments')
#end
	f=0.0
#try
# ignore complex parts
	x = np.real(x)
	y = np.real(y)

# check input arguments
#[m, dx] = size(x)
#size = x.shape
#if np.min(size) != 1:
#  error('Dimensionality of x must be 1')
# if max(m,dx) < 20
#   error('Not enough observations in x (must be > 20)')
# end

#[m, dy] = size(y)
#if min(m,dy) ~= 1
#  error('Dimensionality of y must be 1')
#end
# if max(m,dy) < 20
#   error('Not enough observations in y (must be > 20)')
# end

	if x.size != y.size:
		print "Error1"
#error('Length of x and y must be equal')
#end

	if refMeasure == 1:
		# uniform reference measure
		x = (x - np.min(x)) / (np.max(x) - np.min(x))
		y = (y - np.min(y)) / (np.max(y) - np.min(y))
	elif refMeasure == 2:
		# Gaussian reference measure
		x = (x - np.mean(x)) / np.std(x)
		y = (y - np.mean(y)) / np.std(y)
	else:
		print "Error2"

	if estimator == 1:
		# difference of entropies
		x.sort()
		y.sort()

		n1 = x.size
		hx = 0.0
		for i in range(0,n1-2):
			delta = x[i+1] - x[i]
			if delta:
				hx = hx + np.log(np.abs(delta))
		hx = hx / (n1 - 1) + psi(n1) - psi(1)

		n2 = y.size
		hy = 0.0
		for i in range(0,n2-2):
			delta = y[i+1] - y[i]
			if delta:
				hy = hy + np.log(np.abs(delta))
		hy = hy / (n2 - 1) + psi(n2) - psi(1)

		f = hy - hx
	elif estimator == 2:
		# integral-approximation based estimator
		a = 0.0
		b = 0.0
		m = x.shape[0]
		ind1 = x.argsort()
		ind2 = y.argsort()
		x.sort()
		y.sort()

		for i in range(0,m-2):
			X1 = x[ind1[i]]  
			X2 = x[ind1[i+1]]
			Y1 = y[ind1[i]] 
			Y2 = y[ind1[i+1]]
			if (X2 != X1) and (Y2 != Y1): 
				a = a + np.log(np.abs((Y2 - Y1) / (X2 - X1)))
			X1 = x[ind2[i]]  
			X2 = x[ind2[i+1]]
			Y1 = y[ind2[i]]
			Y2 = y[ind2[i+1]]
			if (Y2 != Y1) and (X2 != X1):
				b = b + np.log(np.abs((X2 - X1) / (Y2 - Y1)))
		f = (a - b)/m
	else: 
		print "Error2"
	return -f # IG
