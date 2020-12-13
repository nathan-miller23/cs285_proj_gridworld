import numpy as np
from utils import softmax, KL_divergence

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

# if __name__ == '__main__':
#     a = np.random.random((10, 3)) * 2 - 1
#     probs = softmax(a)
#     assert np.allclose(np.ones(10), probs.sum(-1))

#     a_preds = np.random.random((10, 3)) * 2 - 1
#     b_preds = (np.random.random((10, 3)) * 2 - 1) * 1e-2 + a
#     c_preds = (np.random.random((10, 3)) * 2 - 1) * 1e-2 + a
#     preds = np.array([a_preds, b_preds, c_preds])
#     info_rad = information_radius_batched(preds)

#     a_prime_preds = np.random.random((10, 3)) * 2 - 1
#     b_prime_preds = (np.random.random((10, 3)) * 2 - 1) + a
#     c_prime_preds = (np.random.random((10, 3)) * 2 - 1) + a
#     preds_prime = np.array([a_prime_preds, b_prime_preds, c_prime_preds])
#     info_rad_prime = information_radius_batched(preds_prime)
#     print(info_rad)
#     print(info_rad_prime)

#     assert info_rad_prime > info_rad    

def train_q_network(ddq_net, expert_data_dict, max_iters=1e4, state_func=False):
    states = expert_data_dict["states"]
    actions = expert_data_dict["actions"] #how do you get this expert_data_dict? look at train.py. its loaded with pickle kk
    rewards = expert_data_dict["rewards"]
    next_states = expert_data_dict["next_states"]
    next_actions = actions[1:] + [0]
    dones = expert_data_dict["dones"]
    
    print("Adding data")
    for s, a, r, s_, a_, d in zip(states, actions, rewards, next_states, next_actions, dones):
        ddq_net.add_data(s, a, r, s_, a_, d)
    print("Finished adding data")
    ddq_net.train(max_iters)

    return ddq_net.a_strat