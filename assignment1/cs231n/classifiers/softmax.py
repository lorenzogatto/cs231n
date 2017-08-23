import numpy as np
from random import shuffle
from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  import math
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  X_len = np.shape(X)[0]
  N_classes = np.shape(W)[1]
  calc = np.dot(X, W)
  for i in xrange(X_len):
    loss -= calc[i, y[i]]
  for i in xrange(X_len):
    for c in xrange(N_classes):
      calc[i, c] = math.exp(calc[i, c])
  sums = np.sum(calc, axis=1)
  for i in xrange(X_len):
    loss += math.log(sums[i])

  loss /= X_len
  loss += reg * np.sum(W*W)


  inverseSum = 1/sums
  upstreamLinear = (calc.T*inverseSum).T
  for i in xrange(X_len):
    for c in xrange(N_classes):
      dW[:, c] += upstreamLinear[i, c] * X[i].T
    dW[:, y[i]] -= X[i].T
  dW /= X_len
  dW += 2 * reg * W;
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  X_len = np.shape(X)[0]
  N_classes = np.shape(W)[1]
  calc = np.dot(X, W)
  #for i in xrange(X_len):
  loss -= np.sum(calc[np.arange(X_len), y])
  calc = np.exp(calc)
  sums = np.sum(calc, axis=1)
  loss += np.sum(np.log(sums))
  loss /= X_len
  loss += reg * np.sum(W*W)
  #inverseSum = 1/sums
  upstreamLinear = (calc.T/sums).T
  dW += np.dot(X.T, upstreamLinear)
  for c in xrange(N_classes):
    dW[:, c] -= np.sum(X[y == c], axis=0)
  #dW[:, :] -= np.sum(X[y], axis=0)
  #dW -=
  #print(np.bincount(y, X.T))
  dW /= X_len
  dW += 2 * reg * W;
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

