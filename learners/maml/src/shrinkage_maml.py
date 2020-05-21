from collections import OrderedDict
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.distributions.normal import Normal
from torch.distributions.categorical import Categorical
from torch.distributions.half_cauchy import HalfCauchy
from torch.distributions.exponential import Exponential
from torch.distributions.kl import kl_divergence
from pytorch.src.utils import LogCauchy, Vague
from pytorch.src.utils import GammaExpMixture as GEM
from pytorch.src.utils import invertible_categorical_kl as ikl_cat

from pytorch.src.maml import MAML


class SigmaMAML(MAML):
    """MAML meta-learner with modular shrinkage. Here implemented with a flat
    prior on sigma. See https://arxiv.org/abs/1909.05557 for details"""
    def __init__(self,
                 update_lr,
                 in_channels,
                 num_classes,
                 pi='vague',
                 beta=0.0001,
                 pi_kwargs={},
                 gradient_steps=1,
                 num_channels=32,
                 kernel_width=3,
                 first_order=True,
                 max_pool=True,
                 flatten=True,
                 hidden_dim=800):
        super(SigmaMAML, self).__init__(update_lr,
                                        in_channels,
                                        num_classes,
                                        gradient_steps,
                                        num_channels,
                                        kernel_width,
                                        first_order,
                                        max_pool,
                                        flatten,
                                        hidden_dim)
        self.num_layers = 5
        # Initialize sigma variables and functions
        self.log_sigmas = self.construct_sigmas()
        self.sigma_fn = nn.Softplus()
        # Regularization distribution
        self.p_eps = Normal(loc=0., scale=1.)

        # Initialize prior on sigma
        self.pi_form = pi
        self._init_pi(pi_kwargs)
        self.beta = beta

    def inner_loop(self, x_context, y_context, num_steps):
        """Run inner loop to get task-specific weights. Use sigmas to
        regularize local updates, and reparameterize local weights via

            phi^m_t = epsilon^m_t * sigma^m + theta^m

        Args:
             x_context (torch.tensor): Context set inputs
             (N x C x H x W)
             y_context (torch.tensor): Context set outputs
             (N x num_classes)
             num_steps (int): Number of gradient steps for optimization

        Returns:
              (dict::params): weight dictionary containing phi_t
        """
        losses = []
        # Initialize epsilon_t
        epsilon = OrderedDict(zip(
            self.weights.keys(),
            [nn.Parameter(torch.zeros_like(self.weights[key]))
             for key in self.weights.keys()]
        ))

        # Enter gradient loop
        for step in range(num_steps):
            # Compute fast weights
            fast_weights = self.reparameterize_weights(epsilon)
            # Forward pass with current weights
            y_ = self(x_context, fast_weights)
            # Compute regularizer and loss function
            regularizer = self.epsilon_regularizer(epsilon)
            loss = F.cross_entropy(y_, y_context) + regularizer

            # Compute gradients and update epsilon
            gradients = self.compute_gradients(loss, epsilon)
            epsilon = OrderedDict(zip(
                epsilon.keys(), 
                [epsilon[key] - self.update_lr * gradients[key]
                 for key in epsilon.keys()]
            ))
            losses.append(loss.data)

        # Final update of weights
        phi_t = self.reparameterize_weights(epsilon)
        return phi_t, losses

    def reparameterize_weights(self, epsilon):
        """Compute fast weights with a given epsilon

        Args:
            epsilon (dict::tensor): dictionary with same structure as
            self.weights containing values of epsilon_t

        Returns:
            (dict::params): weight dictionary containing phi_t
        """
        return OrderedDict(zip(
            self.weights.keys(),
            [self.weights[key] + epsilon[key] * self.extract_sigma(key)
             for key in self.weights.keys()]
        ))

    def epsilon_regularizer(self, epsilon):
        """Compute regularizer for epsilon as N(epsilon; 0, 1)

        Args:
            epsilon (dict::tensor): dictionary with same structure as
            self.weights containing values of epsilon_t

        Returns:
            (torch.scalar): Normal regularization for epsilon
        """
        regularizer = 0
        for key in epsilon.keys():
            regularizer += self.p_eps.log_prob(epsilon[key]).sum()
        return -regularizer

    def construct_sigmas(self):
        """Construct global log-sigma variables, one per layer"""
        sigmas = nn.ParameterDict()

        for layer in range(1, self.num_layers + 1):
            var_name = 'sigma' + str(layer)
            sigmas[var_name] = nn.Parameter(torch.zeros([]))

        return sigmas

    def _init_pi(self, dist_dict):
        """Initialize pi distribution to use over KL values"""
        if self.pi_form == 'vague':
            self.pi = Vague()

        elif self.pi_form == 'cauchy':
            scale = dist_dict.get('scale', 1.)
            self.pi_scale = nn.Parameter(scale * torch.ones([]),
                                         requires_grad=False)
            self.pi = HalfCauchy(scale=self.pi_scale)

        elif self.pi_form == 'log-cauchy':
            scale = dist_dict.get('scale', 2.)
            self.pi = LogCauchy(scale=scale)

        elif self.pi_form == 'exponential':
            rate = dist_dict.get('rate', .5)
            self.pi_rate = nn.Parameter(rate * torch.ones([]),
                                        requires_grad=False)
            self.pi = Exponential(rate=self.pi_rate)

        elif self.pi_form == 'gem':
            learn_mixture = dist_dict.get('learn_mixture', False)
            self.pi = GEM(learn_mixing_rate=learn_mixture)

        else:
            raise ValueError('Unknown pi distribution')

    def sigma_regularizers(self, **kwargs):
        """Compute regularizers for sigma based on PredCP priors.

        Args:
            x (torch.tensor): inputs variables
            (N x C x H x W)
        Returns: 
            (torch.scalar): sum of prior terms for each sigma in model
        """
        return [- self.beta * self.pi.log_prob(self.extract_sigma(key))
                for key in ['conv1', 'conv2', 'conv3', 'conv4']]

    def extract_sigma(self, key):
        """Return the appropriate sigma for a given key of parameters"""
        if '1' in key:
            sigma_key = 'sigma1'
        elif '2' in key:
            sigma_key = 'sigma2'
        elif '3' in key:
            sigma_key = 'sigma3'
        elif '4' in key:
            sigma_key = 'sigma4'
        elif '5' in key:
            sigma_key = 'sigma5'
        else:
            raise ValueError('Key does not exist in model params')
        return self.sigma_fn(self.log_sigmas[sigma_key])

    def print_sigmas(self):
        """Print the values of model sigmas in a list"""
        for key in self.log_sigmas.keys():
            sigma = self.sigma_fn(self.log_sigmas[key]).item()
            print('%s: %s' % (key, sigma)) 


class PredCPMAML(SigmaMAML):
    """MAML meta-learner with modular shrinkage. Implements a PredCP prior on
    sigma (unpublished work)."""
    def __init__(self,
                 update_lr,
                 in_channels,
                 num_classes,
                 pi='half-cauchy',
                 beta=0.0001,
                 pi_kwargs={},
                 gradient_steps=1,
                 num_channels=32,
                 kernel_width=3,
                 first_order=True,
                 max_pool=True,
                 flatten=True,
                 hidden_dim=800):
        super(PredCPMAML, self).__init__(update_lr,
                                         in_channels,
                                         num_classes,
                                         pi=pi,
                                         beta=beta,
                                         pi_kwargs=pi_kwargs,
                                         gradient_steps=gradient_steps,
                                         num_channels=num_channels,
                                         kernel_width=kernel_width,
                                         first_order=first_order,
                                         max_pool=max_pool,
                                         flatten=flatten,
                                         hidden_dim=hidden_dim)

    def sigma_regularizers(self, x, num_samples=1):
        """Compute regularizers for sigma based on PredCP priors. For
        computational efficiency, this happens once per batch, so inputs are
        5-tensors.

        Args:
            x (torch.tensor): inputs variables
            (batch_size x N x C x H x W)
            num_samples (int): number of samples for integration against phi_m
            (Default: 1)
        Returns: 
            (torch.scalar): sum of prior terms for each sigma in model
        """
        # Shape preamble
        batch_size, num_points = x.shape[0], x.shape[1]
        C, H, W = x.shape[2], x.shape[3], x.shape[4]

        # Unroll inputs and compute unnormalized p_0(D | theta)
        x_batch = x.view(-1, C, H, W)
        logits_0 = self(x_batch)
        p0 = Categorical(logits=logits_0)        
        
        # Compute prior term for each sigma
        return [-self.module_regularizer(x, p0, key, num_samples)
                for key in ['conv1', 'conv2', 'conv3', 'conv4']]

    def module_regularizer(self, x, p0, module_key, num_samples):
        """Compute regularizer for module sigma. First, compute an upper bound 
        to KL(p_m || p_0) for a given module m. Second, compute the log
        density of the kl term under pi(KL). Finally, compute change of
        variables term, and return log pi(KL) + log det |J|.

        For computational efficiency, this is computed once per batch.
        Therefore, `x` is a 5-tensor.

        Args:
            x (torch.tensor): inputs variables
            (batch_size x N x C x H x W)
            p0 (torch.distribution): p0 predictive distribution
            (N x num_classes)
            module_key (str): key of module to compute KL 
            num_samples (int): number of samples for integration against phi_m
        Returns:
            (torch.scalar) KL(p_m || p_0 ; D)
        """
        # Shape preamble
        batch_size, num_points = x.shape[0], x.shape[1]
        C, H, W = x.shape[2], x.shape[3], x.shape[4]

        # Sample from p(phi_m | theta, sigma_m)
        sigma = self.extract_sigma(module_key)
        p_phi = Normal(loc=self.weights[module_key], 
                       scale=sigma * torch.ones_like(self.weights[module_key]))
        
        # Helper function for inserting module weights
        def insert_module(module_sample, m_key):
            return  OrderedDict(zip(
                self.weights.keys(),
                [module_sample if key == m_key else self.weights[key]
                 for key in self.weights.keys()]
            ))

        # Unroll `x` and compute p_m(D | theta, phi_m) for every sample in phi
        x_batch = x.view(-1, C, H, W)
        logits_m = torch.stack([self(x_batch, insert_module(phi, module_key)) 
                                for phi in p_phi.rsample([num_samples])])
        pm = Categorical(logits=logits_m)

        # Compute average KL(pm || p0) across samples, and log pi(kl)
        # kl_terms = ikl_cat(logits_m, p0)
        kl_terms = kl_divergence(pm, p0)
        kl_term = kl_terms.view(batch_size, num_points).mean(1)
        log_pi_kl = self.pi.log_prob(kl_term)

        # Compute change of variable term, and return regularizer
        jacobian = torch.stack(
            [torch.autograd.grad(kl, sigma, create_graph=True)[0]
             for kl in kl_term]
        )
        prior_densities = (log_pi_kl + torch.log(torch.abs(jacobian))).sum()
        return self.beta * prior_densities 

