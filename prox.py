import numpy as np
from sklearn.isotonic import isotonic_regression
import torch as th
from torch import nn

def OWL_eval(beta,weights):
    v_abs = np.abs(beta)
    ix = np.argsort(v_abs)[::-1]
    v_abs = v_abs[ix]
    return weights.dot(v_abs)

class OWL_prox(nn.Module):
    def __init__(self,weights):
        super().__init__()
        self.weights=weights

    def forward(self,u,x):
        temp_diff = (u - x).view(-1).cpu().numpy() #flattened
        proxd = proxOWL(temp_diff,self.weights) #needs flattened vector and weights
        proxd_th = th.from_numpy(proxd).cuda() #move back to pytorch
        return x + proxd_th.float() #finish call

def proxOWL(beta, weights):
    p = len(beta)
    abs_beta = np.abs(beta)
    ix = np.argsort(abs_beta)[::-1]
    abs_beta = abs_beta[ix]
    iso_input = abs_beta - weights
    abs_beta = isotonic_regression(iso_input, y_min=0, increasing=False)

    idxs = np.zeros_like(ix)
    idxs[ix] = np.arange(p)
    abs_beta = abs_beta[idxs]

    beta = np.sign(beta) * abs_beta
    return beta


