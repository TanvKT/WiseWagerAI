import torch
import torch.nn as nn
import torch.nn.functional as f
import MACROS
import math

# global constants
ONE_OVER_SQRT2PI = 1.0 / math.sqrt(2 * math.pi)


class MDN(nn.Module):
    def __init__(self, input_size, output_size, num_dist, dist='gaussian'):
        """
        Mixture Density Model
        Predict underlying distributions of data for better uncertainty relations
        Logic behind this is that we are delivering a risk rating, and understanding the uncertainty
            within our prediction allows for a more accurate rating
        Also allows for more general statistical analysis to be done on network outputs
        Mostly inspired by pytorch-mdn input by sagelywizard:
            https://github.com/sagelywizard/pytorch-mdn/blob/master/

        :param input_size: Number of inputs to network
        :param output_size: Number of outputs to network
        :param num_dist: Number of distributions to output
        :param dist: Default gaussian, defines output distribution type
        """
        super(MDN, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.num_dist = num_dist
        self.dist = dist
        self.pi = nn.Sequential(
            nn.Linear(input_size, num_dist),
            nn.Softmax(dim=1)
        )

        ###################################################################
        # Add different distributions as necessary
        ###################################################################
        if dist == 'gaussian' or dist == 'laplace':
            self.sigma = nn.Linear(input_size, output_size * num_dist)
        elif dist == 'poisson':
            self.sigma = None
        else:
            print("Undefined distribution, Defaulting to gaussian")
            self.sigma = nn.Linear(input_size, output_size * num_dist)

        self.mu = nn.Linear(input_size, output_size * num_dist)

    def forward(self, x):
        """
        mu is the average of each distribution in the batch
        sigma is the standard deviation of each distribution in the batch
        pi is the multinomial distribution of the distributions
        :param x: inputs in batch
        :return: Predicted output distributions
        """
        pi = self.pi(x)
        mu = self.mu(x)
        if self.dist == 'poisson':
            mu = torch.exp(mu)
        mu = mu.view(-1, self.num_dist, self.output_size)
        if self.sigma is not None:
            sigma = torch.exp(self.sigma(x))
            sigma = sigma.view(-1, self.num_dist, self.output_size)
        else:
            sigma = torch.sqrt(mu)
        return pi, sigma, mu


# LSTM-MDN Model architecture
class NeuralNet(nn.Module):
    def __init__(self, current_size, input_size, hidden_size, output_size, num_layers, num_dist, dist='gaussian',
                 dropout_prob=0.5):
        """
        Simple LSTM Model for training on specific player
        :param team_size: Number of features in current game (I.E home or away/team rank)
        :param input_size: Number of features to input
        :param hidden_size: Number of hidden states in LSTM
        :param output_size: Number of outputs
        :param num_layers: Number of layers in LSTM
        :param dropout_prob: Parameter to help with over-fitting
        """
        super(NeuralNet, self).__init__()
        self.current_size = current_size
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = nn.Dropout(p=dropout_prob)
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.batch_norm = nn.BatchNorm1d(hidden_size + current_size)  # Batch normalization layer
        self.norm = nn.LayerNorm([1, hidden_size + current_size])
        self.mdn = MDN(hidden_size + current_size, output_size, num_dist, dist)

    def forward(self, x):
        """
        Forward propagate using normalization techniques to help with over-fitting
        :param x: inputs in batch
                    Assume -> team statistics are given first, then lstm sequential game statistics
        :return: MDN output of LSTM prediction
        """
        # format input tensors
        # isolate first current_size values from each batch
        curr, prev = torch.split(x, [self.current_size, self.input_size * MACROS.time_steps], 1)
        curr = curr.reshape(curr.shape[0], self.current_size)
        prev = prev.reshape(prev.shape[0], MACROS.time_steps, self.input_size)

        #############################################################################
        # PICK WHICH ONE BASED ON IF YOU WANT TO KEEP HIDDEN AND CELL STATES

        # zero out desired states
        hidden_states = torch.zeros(self.num_layers, x.shape[0], self.hidden_size)
        cell_states = torch.zeros(self.num_layers, x.shape[0], self.hidden_size)

        # compute output based on specific states
        # out, _ = self.lstm(x)
        out, _ = self.lstm(prev, (hidden_states, cell_states))
        #############################################################################

        # Apply dropout
        # out = self.dropout(out)

        # attach current data to out
        out = out[:, -1, :]
        out = torch.cat((out, curr), dim=1)

        # Apply batch normalization
        if out.shape[0] != 1:
            out = self.batch_norm(out)
        else:
            out = self.norm(out)

        # Apply activation function
        out = f.tanh(out)

        # Apply MDN
        out = self.mdn(out)
        return out

    def backward(self, retain_variables=True):
        self.loss.backward(retain_variables=retain_variables)
        return self.loss


def gaussian_probability(sigma, mu, target):
    """
    Returns the probability of `target` given MoG parameters `sigma` and `mu`.

    Arguments:
        sigma (BxGxO): The standard deviation of the Gaussians. B is the batch
            size, G is the number of Gaussians, and O is the number of
            dimensions per Gaussian.
        mu (BxGxO): The means of the Gaussians. B is the batch size, G is the
            number of Gaussians, and O is the number of dimensions per Gaussian.
        target (BxI): A batch of target. B is the batch size and I is the number of
            input dimensions.

    Returns:
        probabilities (BxG): The probability of each point in the probability
            of the distribution in the corresponding sigma/mu index.
    """
    ret = torch.exp(-0.5 * torch.pow(((target - mu) / sigma), 2))
    return ONE_OVER_SQRT2PI * ret / sigma


def poisson_probability(lam, target):
    """
    Uses the density function for a poisson dist given lam to find probability
    Important to none that since we are dealing with values that can possibly be zero,
        every value is shifted up by 1.0 i.e. zero shots on target is viewed as one shot on target for the prediction
    :param lam: lambda value
    :param target: desired output
    :return: probability of target in desired distribution
    """

    # poisson distribution using torch functions
    return (torch.exp(-lam) * torch.pow(lam, target)) / torch.exp(torch.lgamma(target + 1))


def laplace_probability(mu, sigma, target):
    """
    Uses the laplace density function given input parameters to determine probability of target falling
        within distribution
    Assume sigma is never 0
    :param mu: expected value
    :param sigma: standard deviation
    :param target: desired output
    :return: probability of target in desired distribution
    """

    # find b value
    b = torch.sqrt(0.5 * sigma)

    # laplace distribution using torch functions
    return (1/(2*b)) * torch.exp(-(torch.abs(target - mu)/b))


def loss(mu, sigma, pi, labels, dist='gaussian'):
    """
    Calculate error given distribution parameters
    :param mu: Average value of distribution in batch
    :param sigma: Standard deviation of distribution in batch
    :param pi: Multinomial distribution of distribution in batch
    :param labels: Expected values of output
    :param dist: Distribution type, default gaussian
    :return: torch loss-type for use in optimization
    """
    if dist == 'gaussian':
        # ensure same shape
        labels = labels.unsqueeze(1).expand_as(sigma)
        # product if given multiple values
        prob = pi * torch.prod(gaussian_probability(sigma, mu, labels), 2)
    elif dist == 'poisson':
        labels = labels.unsqueeze(1).expand_as(mu)
        labels = labels + 1
        prob = pi * torch.prod(poisson_probability(mu, labels), 2)
    elif dist == 'laplace':
        labels = labels.unsqueeze(1).expand_as(mu)
        prob = pi * torch.prod(laplace_probability(mu, sigma, labels), 2)
    else:
        print("Undefined distribution, Defaulting to gaussian")
        labels = labels.unsqueeze(1).expand_as(sigma)
        prob = pi * torch.prod(gaussian_probability(sigma, mu, labels), 2)

    s = torch.sum(prob, dim=1)
    for i in range(len(s)):
        if s[i] == 0:
            s[i] = 0.00000001  # ensure no invalid values for log loss function
    nll = -torch.log(s)
    return torch.mean(nll)


def save(model, path: str):
    """
    Saves network
    :param model: model to save
    :param path: path to save model
    :return: Nothing
    """
    torch.save(model.state_dict(), path)
