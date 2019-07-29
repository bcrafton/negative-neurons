
import numpy as np

def softmax(x):
    e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e_x / np.sum(e_x, axis=1, keepdims=True)

def sigmoid(x):
  return 1. / (1. + np.exp(-x))
  
def dsigmoid(x):
  # USE A NOT Z ???
  return x * (1. - x)

def relu(x):
  return x * (x > 0)
  
def drelu(x):
  # USE A NOT Z
  return 1.0 * (x > 0)
