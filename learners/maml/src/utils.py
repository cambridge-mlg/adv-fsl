import torch
import torch.nn as nn
import numpy as np

from torch.distributions.exponential import Exponential
from torch.distributions.gamma import Gamma
from torch.distributions.half_cauchy import HalfCauchy
from torch.distributions.cauchy import Cauchy

from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class GammaExpMixture(nn.Module):
    """Simple wrapper for mixture distribution of Gamma and Exponential"""
    def __init__(self,
                 learn_mixing_rate=False,
                 g_concentration=.2,
                 g_rate=2.,
                 e_rate=.5):
        super(GammaExpMixture, self).__init__()
        self.learn_mixing_rate = learn_mixing_rate

        # Initialize Gamma distribution
        self.gamma_a = nn.Parameter(g_concentration * torch.ones([]),
                                    requires_grad=False)
        self.gamma_b = nn.Parameter(g_rate * torch.ones([]),
                                    requires_grad=False)
        self.gamma = Gamma(concentration=self.gamma_a, rate=self.gamma_b)

        # Initialize Exponential distribution
        self.exp_rate = nn.Parameter(e_rate * torch.ones([]),
                                     requires_grad=False)
        self.exponential = Exponential(rate=self.exp_rate)

        # Initialize mixing rate
        self.mix_rate = nn.Parameter(0.5 * torch.ones([]),
                                     requires_grad=self.learn_mixing_rate)

    def forward(self, x):
        """Dummy method for forward pass"""
        return x

    def log_prob(self, x):
        """Compute log-density under mixture distribution"""
        gamma_log_prob = self.gamma.log_prob(x)
        exp_log_prob = self.exponential.log_prob(x)
        return self.mix_rate * gamma_log_prob + \
               (1 - self.mix_rate) * exp_log_prob


class LogCauchy(nn.Module):
    """Simple wrapper for log-Cauchy distribution"""
    def __init__(self, scale):
        super(LogCauchy, self).__init__()

        # Initialize half-Cauchy distribution
        self.scale = nn.Parameter(scale * torch.ones([]), requires_grad=False)
        self.loc = nn.Parameter(torch.zeros([], requires_grad=False))
        self.cauchy = Cauchy(loc=self.loc, scale=self.scale)

    def forward(self, x):
        """Dummy method for forward pass"""
        return x

    def log_prob(self, x):
        """Compute log-density under log-Cauchy distribution"""
        return self.cauchy.log_prob(torch.log(x))


class Vague(nn.Module):
    """Simple wrapper for 'vague' prior"""
    def __init__(self):
        super(Vague, self).__init__()
        self.val = nn.Parameter(torch.tensor([0]), requires_grad=False)

    def forward(self, x):
        """Dummy method for forward pass"""
        return x

    def log_prob(self, x):
        """Compute log-density under log-Cauchy distribution"""
        return self.val


""" Invertible Categorical Distributions """


def invertible_categorical_kl(p_logits, q_logits):
    """ Compute an invertible pseudo-KL(p||q) divergence between to Categorical
    distributions.
    
    Args:
        p_logits (torch.tensor): logits of p distribution
        (... x dim_p)
        q_logits (torch.tensor): logits of q distribution
        (... x dim_q)

    Returns:
        (torch.tensor) invertible kl(p||q)
        (... x 1)
    """
    assert p_logits.shape[-1] == q_logits.shape[-1]

    p_logits = invertible_log_softmax(p_logits, dim=-1, epsilon=-100)
    q_logits = invertible_log_softmax(q_logits, dim=-1, epsilon=-100)

    # Compute softmax probabilities with constant on normalizer
    p_probs = torch.exp(p_logits)
    t = p_probs * (p_logits - q_logits)
    return t.sum(-1)


def mykl(p, q):
    p_logits = nn.functional.log_softmax(p, dim=-1)
    q_logits = nn.functional.log_softmax(q, dim=-1)

    p_probs = torch.exp(p_logits)
    t = p_probs * (p_logits - q_logits)
    return t.sum(-1)


def invertible_log_softmax(logits, dim=-1, epsilon=-2.):
    epsilon = epsilon * torch.ones(*logits.shape[:-1], 1).to(device)
    logits_ = torch.cat([logits, epsilon], dim=dim)
    return logits - logits_.logsumexp(dim, keepdim=True)
    

def cross_entropy(y, y_target, reduction=None):
    """Testing cross entropy implementation to verify correctness"""
    y_ = y.index_select(-1, y_target).diag()
    entropies = -y_ + torch.logsumexp(y, dim=-1)

    if reduction is None or reduction == 'mean':
        return entropies.mean()

    elif reduction == 'sum':
        return entropies.sum()

    elif reduction == 'none':
        return entropies 

    else:
        raise ValueError('Reduction type invalid')


def invertible_cross_entropy_v2(y, y_target, reduction=None):
    """Modify softmax cross entropy by adding a 1 to logsumexp"""
    # Extract relevant logit
    y_ = y.index_select(-1, y_target).diag()

    # Append zeros to logits
    zeroes = torch.zeros(y.shape[0], 1).to(device)
    tilde_y = torch.cat([y, zeroes], dim=-1)
    entropies = -y_ + torch.logsumexp(tilde_y, dim=-1)

    # Reduce and return
    if reduction is None or reduction == 'mean':
        return entropies.mean()

    elif reduction == 'sum':
        return entropies.sum()

    elif reduction == 'none':
        return entropies 

    else:
        raise ValueError('Reduction type invalid')


def invertible_cross_entropy(y, y_target, reduction=None):
    """Modify softmax cross entropy by adding a 1 to logsumexp"""
    # Append zeros to logits
    zeroes = torch.zeros(y.shape[0], 1).to(device)
    tilde_y = torch.cat([y, zeroes], dim=-1)
    return nn.functional.cross_entropy(tilde_y, y_target, reduction=reduction)


def test_invertible_ce(y_, y_target):
    ce1 = invertible_cross_entropy(y_, y_target, 'sum')
    ce2 = invertible_cross_entropy_v2(y_, y_target, 'sum')
    return torch.isclose(ce1, ce2)


def save_image(image_array, save_path):
    image_array = image_array.transpose([1, 2, 0])
    mode = 'RGB'
    if image_array.shape[2] == 1:  # single channel image
        image_array = image_array.squeeze()
        mode = 'L'
    im = Image.fromarray(np.clip(image_array * 255.0, 0, 255).astype(np.uint8), mode=mode)
    im.save(save_path)
