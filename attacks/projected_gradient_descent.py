import torch.nn as nn
import torch
import math
import torch.distributions.uniform as uniform
import numpy as np
from attacks.attack_utils import convert_labels


class ProjectedGradientDescent:
    def __init__(self,
                 norm='inf',
                 epsilon=0.3,
                 num_iterations=10,
                 epsilon_step=0.01,
                 project_step=True,
                 attack_mode='context',
                 class_fraction=1.0,
                 shot_fraction=1.0,
                 clip_max=1.0,
                 clip_min=-1.0):
        self.norm = norm
        self.epsilon = epsilon
        self.num_iterations = num_iterations
        self.epsilon_step = epsilon_step
        self.project_step = project_step
        self.attack_mode = attack_mode
        self.class_fraction = class_fraction
        self.shot_fraction = shot_fraction
        self.clip_max = clip_max
        self.clip_min = clip_min
        self.loss = nn.CrossEntropyLoss()

    def generate(self, context_images, context_labels, target_images, model):
        adv_context_indices = self._generate_context_attack_indices(context_labels)
        adv_context_images = context_images.clone()

        # Initial projection step
        size = adv_context_images.size()
        m = size[1] * size[2] * size[3]
        initial_perturb = self.random_sphere(len(adv_context_indices), m, self.epsilon, self.norm).reshape((len(adv_context_indices), size[1], size[2], size[3])).to(model.device)

        for i, index in enumerate(adv_context_indices):
            adv_context_images[index] = torch.clamp(adv_context_images[index] +
                                                     initial_perturb[i], self.clip_min,
                                                    self.clip_max)

        adv_context_images.requires_grad = True
        for i in range(0, self.num_iterations):
            logits = model(adv_context_images, context_labels, target_images)
            labels = convert_labels(logits[0])
            # compute loss
            loss = self.loss(logits[0], labels)
            model.zero_grad()

            # compute gradients
            loss.backward()
            grad = adv_context_images.grad

            # apply norm bound
            if self.norm == 'inf':
                perturbation = torch.sign(grad)

            for index in adv_context_indices:
                adv_context_images[index] = torch.clamp(adv_context_images[index] +
                                                        self.epsilon_step * perturbation[index], self.clip_min, self.clip_max)

                diff = adv_context_images[index] - context_images[index]
                new_perturbation = self.projection(diff, self.epsilon, self.norm, model.device)
                adv_context_images[index] = context_images[index] + new_perturbation

            adv_context_images = adv_context_images.detach()
            adv_context_images.requires_grad = True
            del logits

        return adv_context_images, adv_context_indices

    @staticmethod
    def projection(values, eps, norm_p, device):
        """
        Project `values` on the L_p norm ball of size `eps`.
        :param values: Array of perturbations to clip.
        :type values: `pytorch.Tensor`
        :param eps: Maximum norm allowed.
        :type eps: `float`
        :param norm_p: L_p norm to use for clipping. Only 1, 2 and `np.Inf` supported for now.
        :type norm_p: `int`
        :return: Values of `values` after projection.
        :rtype: `pytorch.Tensor`
        """
        # Pick a small scalar to avoid division by 0
        tol = 10e-8
        values_tmp = values.reshape((values.shape[0], -1))

        if norm_p == 2:
            pass
            #values_tmp = values_tmp * torch.unsqueeze(
            #    np.minimum(1.0, eps / (np.linalg.norm(values_tmp, axis=1) + tol)), dim=1
            #)
        elif norm_p == 1:
            pass
            #values_tmp = values_tmp * np.expand_dims(
            #    np.minimum(1.0, eps / (np.linalg.norm(values_tmp, axis=1, ord=1) + tol)), axis=1
            #)
        elif norm_p == 'inf':
            values_tmp = torch.sign(values_tmp) * torch.min(torch.abs(values_tmp), torch.Tensor([eps]).to(device))
        else:
            raise NotImplementedError(
                "Values of `norm_p` different from 1, 2 and `np.inf` are currently not supported.")

        values = values_tmp.reshape(values.shape)
        return values

    @staticmethod
    def random_sphere(nb_points, nb_dims, radius, norm):
        """
        Generate randomly `m x n`-dimension points with radius `radius` and centered around 0.
        :param nb_points: Number of random data points
        :type nb_points: `int`
        :param nb_dims: Dimensionality
        :type nb_dims: `int`
        :param radius: Radius
        :type radius: `float`
        :param norm: Current support: 1, 2, np.inf
        :type norm: `int`
        :return: The generated random sphere
        :rtype: `np.ndarray`
        """
        if norm == 1:
            '''
            a_tmp = np.zeros(shape=(nb_points, nb_dims + 1))
            a_tmp[:, -1] = np.sqrt(np.random.uniform(0, radius ** 2, nb_points))

            for i in range(nb_points):
                a_tmp[i, 1:-1] = np.sort(np.random.uniform(0, a_tmp[i, -1], nb_dims - 1))

            res = (a_tmp[:, 1:] - a_tmp[:, :-1]) * np.random.choice([-1, 1], (nb_points, nb_dims))
            '''
        elif norm == 2:
            '''
            # pylint: disable=E0611
            from scipy.special import gammainc

            a_tmp = np.random.randn(nb_points, nb_dims)
            s_2 = np.sum(a_tmp ** 2, axis=1)
            base = gammainc(nb_dims / 2.0, s_2 / 2.0) ** (1 / nb_dims) * radius / np.sqrt(s_2)
            res = a_tmp * (np.tile(base, (nb_dims, 1))).T
            '''
        elif norm == 'inf':
            distr = uniform.Uniform(torch.Tensor([-radius]), torch.Tensor([radius]))
            res = distr.sample((nb_points, nb_dims))
        else:
            raise NotImplementedError("Norm {} not supported".format(norm))

        return res

