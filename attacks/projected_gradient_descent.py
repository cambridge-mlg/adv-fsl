import torch.nn as nn
import torch
import math
import torch.distributions.uniform as uniform
import numpy as np
from matplotlib import pyplot as plt
import os.path as path
from attacks.attack_utils import convert_labels, generate_attack_indices, fix_logits, Logger
from attacks.attack_utils import get_shifted_targeted_labels, get_random_targeted_labels


class ProjectedGradientDescent:
    def __init__(self,
                 checkpoint_dir,
                 norm='inf',
                 epsilon=0.3,
                 num_iterations=10,
                 epsilon_step=0.01,
                 project_step=True,
                 attack_mode='context',
                 class_fraction=1.0,
                 shot_fraction=1.0,
                 use_true_target_labels=False,
                 normalize_perturbation=True,
                 target_loss_mode='all',
                 targeted=False,
                 targeted_labels='random'):
        self.norm = norm
        self.epsilon = epsilon
        self.num_iterations = num_iterations
        self.epsilon_step = epsilon_step
        self.project_step = project_step
        self.attack_mode = attack_mode
        self.class_fraction = class_fraction
        self.shot_fraction = shot_fraction
        # Whether to use true or predicted labels when generating the attack
        # Note that this is independent of whether the attack is targeted or not.
        # We could use either true or predicted labels when selecting "target attack" labels for a targeted attack
        self.use_true_target_labels = use_true_target_labels
        self.normalize_perturbation = normalize_perturbation
        assert target_loss_mode == 'all' or target_loss_mode == 'round_robin' or target_loss_mode == 'random'
        self.target_loss_mode = target_loss_mode
        self.targeted = targeted
        if self.targeted:
            assert targeted_labels == 'exact' or targeted_labels == 'random' or targeted_labels == 'shifted'
        self.targeted_labels = targeted_labels

        self.loss = nn.CrossEntropyLoss()
        self.logger = Logger(checkpoint_dir, "pgd_logs.txt")

        self.verbose = False

    # Epsilon and epsilon_step are specified for inputs normalized to [0,1].
    # Use a sample of the images to recalculate the required perturbation size (for actual image normalization)
    def normalize_epsilon(self, clip_min, clip_max):
        epsilon_new = self.epsilon
        epsilon_step_new = self.epsilon_step
        if clip_min != 0.0 or clip_max != 1.0:
            # Epsilon is specified relative to min = 0, max = 1.0
            # If this is not the case, scale epsilon
            step_ratio = self.epsilon_step / self.epsilon
            epsilon_new = self.epsilon * (clip_max - clip_min)
            epsilon_step_new = epsilon_new * step_ratio
        return epsilon_new, epsilon_step_new

    def generate(self, context_images, context_labels, target_images, true_target_labels, model, get_logits_fn, device, targeted_labels=None):
        if self.use_true_target_labels:
            labels = true_target_labels
        else:
            # get the predicted target labels
            logits = fix_logits(get_logits_fn(context_images, context_labels, target_images))
            labels = convert_labels(logits)

        if self.targeted and targeted_labels is None:
            if self.targeted_labels == 'exact':
                # When in 'exact' mode, the actual targeted labels must be specified
                assert targeted_labels is not None
            elif self.targeted_labels == 'shifted':
                self.logger.log("Generating shifted labels for targeted attack")
                targeted_labels = get_shifted_targeted_labels(true_target_labels, device)
            else: #random labels
                self.logger.print_and_log("Generating random labels for targeted attack")
                targeted_labels = get_random_targeted_labels(true_target_labels, device)

        self.logger.log("Performing PGD attack on {} set. Settings = (norm={}, epsilon={}, epsilon_step={}, "
                        "num_iterations={}, use_true_labels={},target_loss_mode={})"
                        .format(self.attack_mode, self.norm, self.epsilon, self.epsilon_step,
                                self.num_iterations, self.use_true_target_labels, self.target_loss_mode))
        self.logger.log("class_fraction = {}, shot_fraction = {}".format(self.class_fraction, self.shot_fraction))
        self.logger.log("context set size = {}, target set size = {}".format(context_images.shape[0], target_images.shape[0]))

        if self.attack_mode == 'target':
            return self._generate_target(context_images, context_labels, target_images, labels, model, get_logits_fn,
                                         device, targeted_labels=targeted_labels)
        else:  # context
            return self._generate_context(context_images, context_labels, target_images, labels, model, get_logits_fn,
                                          device, targeted_labels=targeted_labels)

    def _generate_target(self, context_images, context_labels, target_images, target_labels, model, get_logits_fn, device, targeted_labels=None):
        clip_min = target_images.min().item()
        clip_max = target_images.max().item()
        # Desired epsilon are given for images normalized to [0,1]. Adjust for when this is not the case
        if self.normalize_perturbation:
            epsilon, epsilon_step = self.normalize_epsilon(clip_min, clip_max)
        else:
            epsilon, epsilon_step = self.epsilon, self.epsilon_step
        self.logger.print_and_log("Normalized perturbation sizes: eps={}, eps_step={}".format(epsilon, epsilon_step))

        if self.targeted:
            labels = targeted_labels # As in the labels we want the target set to be classified as, to be used as "targets"
        else:
            labels = target_labels # As in, the true/predicted labels for the target set

        adv_target_indices = generate_attack_indices(target_labels, self.class_fraction, self.shot_fraction)
        adv_target_images = target_images.clone()

        # Initial projection step
        target_size = adv_target_images.size()
        m = target_size[1] * target_size[2] * target_size[3]
        num_target_images = target_size[0]
        initial_perturb = self.random_sphere(len(adv_target_indices), m, epsilon, self.norm).reshape(
            (len(adv_target_indices), target_size[1], target_size[2], target_size[3])).to(device)

        for i, index in enumerate(adv_target_indices):
            adv_target_images[index] = torch.clamp(adv_target_images[index] + initial_perturb[i], clip_min, clip_max)

        if self.verbose:
            verbose_result = ProjectedGradientDescent.make_verbose_PGD_result()
            verbose_result['adv_images'].append(adv_target_images.clone().detach())

        for i in range(self.num_iterations):
            adv_target_images.requires_grad = True
            logits = fix_logits(get_logits_fn(context_images, context_labels, adv_target_images))
            if self.verbose:
                verbose_result['target_logits'].append(logits.clone().detach())
            # compute loss
            loss = self.loss(logits, labels)
            model.zero_grad()

            if self.verbose and i % 5 == 0 or i == self.num_iterations-1:
                self.logger.print_and_log("Iter {}, loss = {:.5f}".format(i, loss))

            # compute gradient
            loss.backward()
            grad = adv_target_images.grad * (1 - 2 * int(self.targeted))

            adv_target_images = adv_target_images.detach()

            # apply norm bound
            if self.norm == 'inf':
                perturbation = torch.sign(grad)

            for index in adv_target_indices:
                adv_target_images[index] = torch.clamp(adv_target_images[index] +
                                                        epsilon_step * perturbation[index],
                                                        clip_min, clip_max)

                diff = adv_target_images[index] - target_images[index]
                new_perturbation = self.projection(diff, epsilon, self.norm, device)
                adv_target_images[index] = target_images[index] + new_perturbation

            if self.verbose:
                verbose_result['adv_images'].append(adv_target_images.clone().detach())
            del logits

        if self.verbose:
            return adv_target_images, adv_target_indices, verbose_result

        return adv_target_images, adv_target_indices

    @staticmethod
    def make_verbose_PGD_result():
        result = {
                  'adv_images': [],
                  'target_logits': [],
                  }
        return result

    def _generate_context(self, context_images, context_labels, target_images, target_labels, model, get_logits_fn, device, targeted_labels=None):
        clip_min = target_images.min().item()
        clip_max = target_images.max().item()
        if self.normalize_perturbation:
            epsilon, epsilon_step = self.normalize_epsilon(clip_min, clip_max)
        else:
            epsilon, epsilon_step = self.epsilon, self.epsilon_step

        adv_context_indices = generate_attack_indices(context_labels, self.class_fraction, self.shot_fraction)
        adv_context_images = context_images.clone()

        if self.targeted:
            labels = targeted_labels # As in the labels we want the target set to be classified as, to be used as "targets"
        else:
            labels = target_labels # As in, the true/predicted labels for the target set

        # Initial projection step
        size = adv_context_images.size()
        m = size[1] * size[2] * size[3]
        initial_perturb = self.random_sphere(len(adv_context_indices), m, epsilon, self.norm).reshape(
            (len(adv_context_indices), size[1], size[2], size[3])).to(device)

        for i, index in enumerate(adv_context_indices):
            adv_context_images[index] = torch.clamp(adv_context_images[index] + initial_perturb[i], clip_min, clip_max)

        if self.verbose:
            verbose_result = ProjectedGradientDescent.make_verbose_PGD_result()
            verbose_result['adv_images'].append(adv_context_images.clone().detach())

        for i in range(0, self.num_iterations):
            adv_context_images.requires_grad = True
            logits = fix_logits(get_logits_fn(adv_context_images, context_labels, target_images))
            if self.verbose:
                verbose_result['target_logits'].append(logits.clone().detach())

            # compute loss
            if self.target_loss_mode == 'round_robin':
                loss = self.loss(logits[i % len(target_images)].unsqueeze(0), labels[i % len(target_images)].unsqueeze(0))
            elif self.target_loss_mode == 'random':
                index = np.random.randint(0, len(target_images))
                loss = self.loss(logits[index].unsqueeze(0), labels[index].unsqueeze(0))
            else:
                loss = self.loss(logits, labels)
            model.zero_grad()

            if self.verbose and i % 5 == 0 or i == self.num_iterations-1:
                acc = torch.mean(torch.eq(target_labels.long(), torch.argmax(logits, dim=-1).long()).float()).item()
                self.logger.print_and_log("Iter {}, loss = {:.5f}, acc = {:.5f}".format(i, loss, acc))

            # compute gradients
            loss.backward()
            # Invert the gradient if the attack is targeted
            grad = adv_context_images.grad * (1 - 2 * int(self.targeted))

            adv_context_images = adv_context_images.detach()

            # apply norm bound
            if self.norm == 'inf':
                perturbation = torch.sign(grad)

            for index in adv_context_indices:
                adv_context_images[index] = torch.clamp(adv_context_images[index] +
                                                        epsilon_step * perturbation[index],
                                                        clip_min, clip_max)

                diff = adv_context_images[index] - context_images[index]
                new_perturbation = self.projection(diff, epsilon, self.norm, device)
                adv_context_images[index] = context_images[index] + new_perturbation

            if self.verbose:
                verbose_result['adv_images'].append(adv_context_images.clone().detach())
            del logits

        if self.verbose:
            return adv_context_images, adv_context_indices, verbose_result

        return adv_context_images, adv_context_indices

    def get_verbose(self):
        return self.verbose

    def set_verbose(self, new_val):
        self.verbose = new_val

    def get_attack_mode(self):
        return self.attack_mode

    def set_attack_mode(self, new_mode):
        assert new_mode == 'context' or new_mode == 'target'
        self.attack_mode = new_mode

    def get_shot_fraction(self):
        return self.shot_fraction

    def set_shot_fraction(self, new_shot_frac):
        assert new_shot_frac <= 1.0 and new_shot_frac >= 0.0
        self.shot_fraction = new_shot_frac

    def get_class_fraction(self):
        return self.class_fraction

    def set_class_fraction(self, new_class_frac):
        assert new_class_frac <= 1.0 and new_class_frac >= 0.0
        self.class_fraction = new_class_frac

    # Some helper functions for gradient debugging
    def _plot_signs(self, grad_signs, img_index, iter,  checkpoint_dir):
        for c in range(0, 3):
            plt.figure()
            plt.imshow(grad_signs[c].cpu(), cmap='hot', interpolation='nearest')
            plt.savefig(path.join(checkpoint_dir, "{}_{}_grad_signs_chan_{}_iter_{:02d}.png".format(self.attack_mode, img_index, c, iter)))
            plt.close()

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

        # if norm_p == 2:
        #    values_tmp = values_tmp * torch.unsqueeze(
        #        np.minimum(1.0, eps / (np.linalg.norm(values_tmp, axis=1) + tol)), dim=1
        #    )
        # elif norm_p == 1:
        #    values_tmp = values_tmp * np.expand_dims(
        #        np.minimum(1.0, eps / (np.linalg.norm(values_tmp, axis=1, ord=1) + tol)), axis=1
        #    )
        if norm_p == 'inf':
            values_tmp = torch.sign(values_tmp) * torch.min(torch.abs(values_tmp), torch.Tensor([eps]).to(device))
        else:
            raise NotImplementedError(
                "Values of `norm_p` different from `np.inf` are currently not supported.")

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
