import torch

import tester
import network
import MACROS
import matplotlib.pyplot as plt

# create tester
model = network.NeuralNet(MACROS.current_size, MACROS.input_size, MACROS.hidden_size, MACROS.output_size, MACROS.num_lstm_layers, MACROS.num_dist, dist=MACROS.dist)
t = tester.Tester(model, is_mdn=True)

# load model
t.load_model(r"Data/laplace-allplayers-20dist")

# load data
t.load_data(r"../data/Train_Data/2021-2022_PL_train.csv")

# predict
(pi, sigma, mu, targets) = t.test_player("Dejan Kulusevski")

# create graphs to understand outputs
for i, target in enumerate(targets):
    # format
    if MACROS.dist == 'gaussian':
        p = network.gaussian_probability(mu[i], sigma[i], target)
    elif MACROS.dist == 'poisson':
        p = network.poisson_probability(mu[i], target)
    elif MACROS.dist == 'laplace':
        p = network.laplace_probability(mu[i], sigma[i], target)
    else:
        p = network.gaussian_probability(mu[i], sigma[i], target)
    p = (torch.squeeze(p, dim=0)).tolist()
    pi[i] = (torch.squeeze(pi[i], dim=0)).tolist()
    sigma[i] = (torch.squeeze(sigma[i], dim=0)).tolist()
    mu[i] = (torch.squeeze(mu[i], dim=0)).tolist()
    target = torch.squeeze(target).item()
    if MACROS.dist == 'poisson':
        target = target + 1

    # plots
    plt.figure(i, figsize=(10, 5))
    ax = plt.subplot(411)
    plt.plot(range(len(pi[i])), pi[i], marker='o', linestyle='-')
    plt.ylabel('pi')
    plt.title('pi - sigma - mu - probability\n'
              'Note that when using poisson dist, target is incremented by one since 0 is a non-valid value')
    plt.subplot(412, sharex=ax)
    plt.plot(range(len(sigma[i])), sigma[i], marker='o', linestyle='-')
    plt.ylabel('sigma')
    plt.subplot(413, sharex=ax)
    plt.plot(range(len(mu[i])), mu[i], marker='o', linestyle='-')
    plt.ylabel('mu')
    plt.subplot(414, sharex=ax)
    plt.plot(range(len(p)), p, marker='o', linestyle='-')
    plt.xlabel(f'target: {target}')
    plt.ylabel('probability')
    plt.show()
