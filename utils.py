import numpy as np
import pickle

def save(data, outfile):
    with open(outfile, 'wb') as f:
        pickle.dump(data, f)

def load(outfile):
    with open(outfile, 'rb') as f:
        return pickle.load(f)

def mannhattan_distance(state_1, state_2):
    return abs(state_1[0] - state_2[0]) + abs(state_1[1] - state_2[1])

def softmax(x):
    x = x.transpose()
    e_x = np.exp(x - np.max(x))
    return (e_x / e_x.sum(axis=0)).transpose()

def KL_divergence(p, q):
    assert len(p.shape) == 1
    assert len(p.shape) == len(q.shape)
    assert len(p) == len(q)
    log_pq = np.log(p/q)
    return np.sum(p * log_pq)