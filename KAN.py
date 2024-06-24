import torch
import torch.nn as nn
import torch.functional as F
import math

def basis_function(x):
    '''
    this is basis function for residual activation functions
    just silu $ x/(1+\exp(-x)) $
    '''
    
    return nn.SiLU(x)

def spline(B, c, x):
    '''
    this is spline function for KNN
    input : list of B function & x & trainable parameter c
    ouput : $ \sum_i c_iB_i(x)$
    '''

    length = B.shape[0]
    sum = 0.0
    for i in range(length):
        sum += c[i] * B[i](x)
    
    return sum

def knn_activation(w_b, w_s, B, c, x):
    '''
    knn activation
    w_b : parameter of basis function
    w_s : parameter of spline function
    B : list of basis function
    c : list of trainable parameter c
    x : input

    $ \phi(x) = w_b*b(x) + w_s*spline(x) $
    '''

    return w_b*basis_function(x) + w_s*spline(B, c, x)


class KANLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(KANLayer, self).__init__()
        '''
        in_features : # of input dim
        out_features : # of output dim
        '''
        self.in_features = in_features
        self.out_features = out_features