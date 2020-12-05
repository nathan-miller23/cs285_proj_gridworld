import numpy as np

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

def information_radius(preds, from_logits=True):
    # shape (ensemble_size, action_dim)
    if from_logits:
        probs = softmax(preds)
    else:
        probs = preds
    
    k, _ = probs.shape
    KLs = []
    mean_probs = probs.mean(0)
    for i in range(k):
        kl = KL_divergence(probs[i, :], mean_probs)
        KLs.append(kl)
    return np.mean(KLs)

def information_radius_batched(ensemble_pred, from_logits=True):
    # Ensemble_pred.shape (ensemble_size, batch_size, action_dim)
    info_radiuses = []
    probs = ensemble_pred.transpose(1, 0, 2)
    for batch in probs:
        info_radiuses.append(information_radius(batch, from_logits))
    return np.mean(info_radiuses)

if __name__ == '__main__':
    a = np.random.random((10, 3)) * 2 - 1
    probs = softmax(a)
    assert np.allclose(np.ones(10), probs.sum(-1))

    a_preds = np.random.random((10, 3)) * 2 - 1
    b_preds = (np.random.random((10, 3)) * 2 - 1) * 1e-2 + a
    c_preds = (np.random.random((10, 3)) * 2 - 1) * 1e-2 + a
    preds = np.array([a_preds, b_preds, c_preds])
    info_rad = information_radius_batched(preds)

    a_prime_preds = np.random.random((10, 3)) * 2 - 1
    b_prime_preds = (np.random.random((10, 3)) * 2 - 1) + a
    c_prime_preds = (np.random.random((10, 3)) * 2 - 1) + a
    preds_prime = np.array([a_prime_preds, b_prime_preds, c_prime_preds])
    info_rad_prime = information_radius_batched(preds_prime)
    print(info_rad)
    print(info_rad_prime)

    assert info_rad_prime > info_rad    
