from builtins import range
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
      score: shape (N,C)
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)
    num_trian = X.shape[0]
    num_class = W.shape[1]

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    for i in range(num_trian):
        scores = X[i].dot(W)
        shift_score = scores - max(scores)
        loss_i = -shift_score[y[i]]+np.log(sum(np.exp(shift_score)))
        loss += loss_i
        for j in range(num_class):
            sofmax_out = np.exp(shift_score[j])/sum(np.exp(shift_score))
            if j == y[i]:
                dW[:,j] += -X[i] + sofmax_out*X[i]
            else:
                dW[:,j] += sofmax_out*X[i]

    loss /= num_trian
    loss +=0.5* reg * np.sum(W * W)
    dW = dW/num_trian + reg* W 
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)
    num_trian = X.shape[0]
    num_class = W.shape[1]

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    score = X.dot(W)
    score = np.exp(score - np.max(score,axis = 1).reshape(-1,1)) 
    score_sum = np.sum(score,axis = 1).reshape(-1,1)
    score /= score_sum
    correct_class_score = score[range(num_trian),y].reshape(num_trian,1)
    loss = np.sum(-np.log(correct_class_score))
    loss /= num_trian
    loss += reg*np.sum(W*W)

    dS = score.copy()
    dS[range(num_trian), list(y)] += -1
    dW = (X.T).dot(dS)
    dW = dW/num_trian + reg* W 
    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
