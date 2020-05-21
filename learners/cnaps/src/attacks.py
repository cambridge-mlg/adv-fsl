import torch.nn as nn
import torch
import math
import torch.distributions.uniform as uniform
from utils import extract_class_indices
import numpy as np


def convert_labels(predictions):
    return torch.argmax(predictions, dim=1, keepdim=False)

class ProjectedGradientDescent():
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

        #Initial projection step
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

    # Potentially cache class distribution later
    def _generate_context_attack_indices(self, class_labels,):
        indices = []
        classes = torch.unique(class_labels)
        num_classes = len(classes)
        num_classes_to_attack = max(1, math.ceil(self.class_fraction * num_classes))
        for c in range(num_classes_to_attack):
            shot_indices = extract_class_indices(class_labels, c)
            num_shots_in_class = len(shot_indices)
            num_shots_to_attack = max(1, math.ceil(self.shot_fraction * num_shots_in_class))
            attack_indices = shot_indices[0:num_shots_to_attack]
            for index in attack_indices:
                indices.append(index)
        return indices

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


class CarliniwWagnerW2():

    def __init__(self,
                 confidence=0.0,
                 learning_rate=0.01,
                 max_iter=10,
                 binary_search_steps=10,
                 initial_const=0.01,
                 max_halving=5,
                 max_doubling=5,
                 attack_mode='context',
                 class_fraction=1.0,
                 shot_fraction=1.0,
                 clip_max=1.0,
                 clip_min=-1.0):
        self.confidence = confidence
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.binary_search_steps = binary_search_steps
        self.initial_const = initial_const
        self.max_halving = max_halving
        self.max_doubling = max_doubling
        self.attack_mode = attack_mode
        self.class_fraction = class_fraction
        self.shot_fraction = shot_fraction
        self.clip_max = clip_max
        self.clip_min = clip_min
        self.loss = nn.CrossEntropyLoss()

        # There are internal hyperparameters:
        # Abort binary search for c if it exceeds this threshold (suggested in Carlini and Wagner (2016)):
        self._c_upper_bound = 10e10
        # Smooth arguments of arctanh by multiplying with this constant to avoid division by zero.
        # It appears this is what Carlini and Wagner (2016) are alluding to in their footnote 8. However, it is not
        # clear how their proposed trick ("instead of scaling by 1/2 we scale by 1/2 + eps") works in detail.
        self._tanh_smoother = 0.999999

    def generate(self, context_images, context_labels, target_images, model):
        """
        Generate adversarial samples and return them in an array.
        :param x: An array with the original inputs to be attacked.
        :type x: `np.ndarray`
        :param y: Target values (class labels) one-hot-encoded of shape (nb_samples, nb_classes) or indices of shape
                  (nb_samples,). If `self.targeted` is true, then `y` represents the target labels. If `self.targeted`
                  is true, then `y_val` represents the target labels. Otherwise, the targets are the original class
                  labels.
        :return: An array holding the adversarial examples.
        :rtype: `np.ndarray`
        """
        adv_context_indices = self._generate_context_attack_indices(context_labels)
        adv_context_images = context_images.clone()
        adv_x_only = torch.index_select(adv_context_images, index=adv_context_indices, dim=0)

        logits = model(context_images, context_labels, target_images)
        y = convert_labels(logits[0])
        one_hot_y = one_hot_embedding(y, model.way)

        # The optimization is performed in tanh space to keep the adversarial images bounded in correct range
        #We're already in range -1 to 1, so I'm not sure we need this.
        adv_x_tanh = original_to_tanh(adv_x_only, self.clip_min, self.clip_max, self._tanh_smoother)

        # Initialize binary search:
        c_current = self.initial_const * torch.ones((adv_x_only.shape[0]), device=model.device)
        c_lower_bound = torch.zeros(adv_x_only.shape[0], device=model.device)
        c_double = torch.ones((adv_x_only.shape[0]), device=model.device) > 0

        # Initialize placeholders for best l2 distance and attack found so far
        best_l2dist =  float("Inf") * torch.ones((adv_x_only.shape[0]), device=model.device)
        best_x_adv_batch = adv_x_only.copy()

        for bss in range(self.binary_search_steps):
            print("Binary search step %i out of %i (c_mean==%f)".format( bss, self.binary_search_steps, np.mean(c_current)))
            nb_active = int(torch.sum(c_current < self._c_upper_bound))
            print("Number of samples with c_current < _c_upper_bound: %i out of %i".format( nb_active, adv_x_only.shape[0]))
            if nb_active == 0:
                break
            learning_rate = self.learning_rate * torch.ones((adv_x_only.shape[0]), device=model.device)

            # Initialize perturbation in tanh space:
            x_adv_batch = adv_x_only.clone()
            x_adv_batch_tanh = adv_x_tanh.clone()

            z_logits, l2dist, loss = self._loss(adv_x_only, x_adv_batch, one_hot_y, c_current)
            attack_success = loss - l2dist <= 0
            overall_attack_success = attack_success

            for i_iter in range(self.max_iter):
                print("Iteration step %i out of %i", i_iter, self.max_iter)
                print("Average Loss: %f", np.mean(loss))
                print("Average L2Dist: %f", np.mean(l2dist))
                print("Average Margin Loss: %f", np.mean(loss - l2dist))
                print(
                    "Current number of succeeded attacks: %i out of %i".format()
                    int(np.sum(attack_success)),
                    len(attack_success),
                )

                improved_adv = attack_success & (l2dist < best_l2dist)
                logger.debug("Number of improved L2 distances: %i", int(np.sum(improved_adv)))
                if np.sum(improved_adv) > 0:
                    best_l2dist[improved_adv] = l2dist[improved_adv]
                    best_x_adv_batch[improved_adv] = x_adv_batch[improved_adv]

                active = (c_current < self._c_upper_bound) & (learning_rate > 0)
                nb_active = int(np.sum(active))
                logger.debug(
                    "Number of samples with c_current < _c_upper_bound and learning_rate > 0: %i out of %i",
                    nb_active,
                    x_batch.shape[0],
                )
                if nb_active == 0:
                    break

                # compute gradient:
                logger.debug("Compute loss gradient")
                perturbation_tanh = -self._loss_gradient(
                    z_logits[active],
                    y_batch[active],
                    x_batch[active],
                    x_adv_batch[active],
                    x_adv_batch_tanh[active],
                    c_current[active],
                    clip_min,
                    clip_max,
                )

                # perform line search to optimize perturbation
                # first, halve the learning rate until perturbation actually decreases the loss:
                prev_loss = loss.copy()
                best_loss = loss.copy()
                best_lr = np.zeros(x_batch.shape[0])
                halving = np.zeros(x_batch.shape[0])

                for i_halve in range(self.max_halving):
                    logger.debug("Perform halving iteration %i out of %i", i_halve, self.max_halving)
                    do_halving = loss[active] >= prev_loss[active]
                    logger.debug("Halving to be performed on %i samples", int(np.sum(do_halving)))
                    if np.sum(do_halving) == 0:
                        break
                    active_and_do_halving = active.copy()
                    active_and_do_halving[active] = do_halving

                    lr_mult = learning_rate[active_and_do_halving]
                    for _ in range(len(x.shape) - 1):
                        lr_mult = lr_mult[:, np.newaxis]

                    x_adv1 = x_adv_batch_tanh[active_and_do_halving]
                    new_x_adv_batch_tanh = x_adv1 + lr_mult * perturbation_tanh[do_halving]
                    new_x_adv_batch = tanh_to_original(
                        new_x_adv_batch_tanh, clip_min, clip_max, self._tanh_smoother
                    )
                    _, l2dist[active_and_do_halving], loss[active_and_do_halving] = self._loss(
                        x_batch[active_and_do_halving],
                        new_x_adv_batch,
                        y_batch[active_and_do_halving],
                        c_current[active_and_do_halving],
                    )

                    logger.debug("New Average Loss: %f", np.mean(loss))
                    logger.debug("New Average L2Dist: %f", np.mean(l2dist))
                    logger.debug("New Average Margin Loss: %f", np.mean(loss - l2dist))

                    best_lr[loss < best_loss] = learning_rate[loss < best_loss]
                    best_loss[loss < best_loss] = loss[loss < best_loss]
                    learning_rate[active_and_do_halving] /= 2
                    halving[active_and_do_halving] += 1
                learning_rate[active] *= 2

                # if no halving was actually required, double the learning rate as long as this
                # decreases the loss:
                for i_double in range(self.max_doubling):
                    logger.debug("Perform doubling iteration %i out of %i", i_double, self.max_doubling)
                    do_doubling = (halving[active] == 1) & (loss[active] <= best_loss[active])
                    logger.debug("Doubling to be performed on %i samples", int(np.sum(do_doubling)))
                    if np.sum(do_doubling) == 0:
                        break
                    active_and_do_doubling = active.copy()
                    active_and_do_doubling[active] = do_doubling
                    learning_rate[active_and_do_doubling] *= 2

                    lr_mult = learning_rate[active_and_do_doubling]
                    for _ in range(len(x.shape) - 1):
                        lr_mult = lr_mult[:, np.newaxis]

                    x_adv2 = x_adv_batch_tanh[active_and_do_doubling]
                    new_x_adv_batch_tanh = x_adv2 + lr_mult * perturbation_tanh[do_doubling]
                    new_x_adv_batch = tanh_to_original(
                        new_x_adv_batch_tanh, clip_min, clip_max, self._tanh_smoother
                    )
                    _, l2dist[active_and_do_doubling], loss[active_and_do_doubling] = self._loss(
                        x_batch[active_and_do_doubling],
                        new_x_adv_batch,
                        y_batch[active_and_do_doubling],
                        c_current[active_and_do_doubling],
                    )
                    logger.debug("New Average Loss: %f", np.mean(loss))
                    logger.debug("New Average L2Dist: %f", np.mean(l2dist))
                    logger.debug("New Average Margin Loss: %f", np.mean(loss - l2dist))
                    best_lr[loss < best_loss] = learning_rate[loss < best_loss]
                    best_loss[loss < best_loss] = loss[loss < best_loss]

                learning_rate[halving == 1] /= 2

                update_adv = best_lr[active] > 0
                logger.debug("Number of adversarial samples to be finally updated: %i", int(np.sum(update_adv)))

                if np.sum(update_adv) > 0:
                    active_and_update_adv = active.copy()
                    active_and_update_adv[active] = update_adv
                    best_lr_mult = best_lr[active_and_update_adv]
                    for _ in range(len(x.shape) - 1):
                        best_lr_mult = best_lr_mult[:, np.newaxis]

                    x_adv4 = x_adv_batch_tanh[active_and_update_adv]
                    best_lr1 = best_lr_mult * perturbation_tanh[update_adv]
                    x_adv_batch_tanh[active_and_update_adv] = x_adv4 + best_lr1

                    x_adv6 = x_adv_batch_tanh[active_and_update_adv]
                    x_adv_batch[active_and_update_adv] = tanh_to_original(
                        x_adv6, clip_min, clip_max, self._tanh_smoother
                    )
                    (
                        z_logits[active_and_update_adv],
                        l2dist[active_and_update_adv],
                        loss[active_and_update_adv],
                    ) = self._loss(
                        x_batch[active_and_update_adv],
                        x_adv_batch[active_and_update_adv],
                        y_batch[active_and_update_adv],
                        c_current[active_and_update_adv],
                    )
                    attack_success = loss - l2dist <= 0
                    overall_attack_success = overall_attack_success | attack_success

            # Update depending on attack success:
            improved_adv = attack_success & (l2dist < best_l2dist)
            logger.debug("Number of improved L2 distances: %i", int(np.sum(improved_adv)))

            if np.sum(improved_adv) > 0:
                best_l2dist[improved_adv] = l2dist[improved_adv]
                best_x_adv_batch[improved_adv] = x_adv_batch[improved_adv]

            c_double[overall_attack_success] = False
            c_current[overall_attack_success] = (c_lower_bound + c_current)[overall_attack_success] / 2

            c_old = c_current
            c_current[~overall_attack_success & c_double] *= 2

            c_current1 = (c_current - c_lower_bound)[~overall_attack_success & ~c_double]
            c_current[~overall_attack_success & ~c_double] += c_current1 / 2
            c_lower_bound[~overall_attack_success] = c_old[~overall_attack_success]

        x_adv[batch_index_1:batch_index_2] = best_x_adv_batch

        logger.info(
            "Success rate of C&W L_2 attack: %.2f%%",
            100 * compute_success(self.classifier, x, y, x_adv, self.targeted, batch_size=self.batch_size),
        )

        return x_adv

    def _loss(self, x, x_adv, target, c_weight, logits):
        """
        Compute the objective function value.
        :param x: An array with the original input.
        :type x: `torch.Tensor`
        :param x_adv: An array with the adversarial input.
        :type x_adv: `torch.Tensor`
        :param target: An array with the target class (one-hot encoded).
        :type target: `torch.Tensor`
        :param c_weight: Weight of the loss term aiming for classification as target.
        :type c_weight: `float`
        :return: A tuple holding the current logits, l2 distance and overall loss.
        :rtype: `(float, float, float)`
        """
        l2dist = torch.sum(torch.square(x - x_adv).reshape(x.shape[0], -1), dim=1)
        z_predicted = logits
        z_target = torch.sum(z_predicted * target, dim=1) #z_predicted * target = logit of correct class
        z_other = torch.max(z_predicted * (1 - target) + (torch.min(z_predicted, dim=1) - 1).view(-1, 1) * target, dim=1)

        # The following differs from the exact definition given in Carlini and Wagner (2016). There (page 9, left
        # column, last equation), the maximum is taken over Z_other - Z_target (or Z_target - Z_other respectively)
        # and -confidence. However, it doesn't seem that that would have the desired effect (loss term is <= 0 if and
        # only if the difference between the logit of the target and any other class differs by at least confidence).
        # Hence the rearrangement here.

        # if untargeted, optimize for making any other class most likely
        loss = torch.clamp(z_target - z_other + self.confidence, min=0.)

        return z_predicted, l2dist, c_weight * loss + l2dist

    def _loss_gradient(self, z_logits, target, x, x_adv, x_adv_tanh, c_weight, clip_min, clip_max):
        """
        Compute the gradient of the loss function.
        :param z_logits: An array with the current logits.
        :type z_logits: `np.ndarray`
        :param target: An array with the target class (one-hot encoded).
        :type target: `np.ndarray`
        :param x: An array with the original input.
        :type x: `np.ndarray`
        :param x_adv: An array with the adversarial input.
        :type x_adv: `np.ndarray`
        :param x_adv_tanh: An array with the adversarial input in tanh space.
        :type x_adv_tanh: `np.ndarray`
        :param c_weight: Weight of the loss term aiming for classification as target.
        :type c_weight: `float`
        :param clip_min: Minimum clipping value.
        :type clip_min: `float`
        :param clip_max: Maximum clipping value.
        :type clip_max: `float`
        :return: An array with the gradient of the loss function.
        :type target: `np.ndarray`
        """
        if self.targeted:
            i_sub = np.argmax(target, axis=1)
            i_add = np.argmax(z_logits * (1 - target) + (np.min(z_logits, axis=1) - 1)[:, np.newaxis] * target, axis=1)
        else:
            i_add = np.argmax(target, axis=1)
            i_sub = np.argmax(z_logits * (1 - target) + (np.min(z_logits, axis=1) - 1)[:, np.newaxis] * target, axis=1)

        loss_gradient = self.classifier.class_gradient(x_adv, label=i_add)
        loss_gradient -= self.classifier.class_gradient(x_adv, label=i_sub)
        loss_gradient = loss_gradient.reshape(x.shape)

        c_mult = c_weight
        for _ in range(len(x.shape) - 1):
            c_mult = c_mult[:, np.newaxis]

        loss_gradient *= c_mult
        loss_gradient += 2 * (x_adv - x)
        loss_gradient *= clip_max - clip_min
        loss_gradient *= (1 - np.square(np.tanh(x_adv_tanh))) / (2 * self._tanh_smoother)

        return loss_gradient

def original_to_tanh(x_original, clip_min, clip_max, tanh_smoother=0.999999):
    """
    Transform input from original to tanh space.
    :param x_original: An array with the input to be transformed.
    :type x_original: `np.ndarray`
    :param clip_min: Minimum clipping value.
    :type clip_min: `float` or `np.ndarray`
    :param clip_max: Maximum clipping value.
    :type clip_max: `float` or `np.ndarray`
    :param tanh_smoother: Scalar for multiplying arguments of arctanh to avoid division by zero.
    :type tanh_smoother: `float`
    :return: An array holding the transformed input.
    :rtype: `np.ndarray`
    """
    x_tanh = torch.clamp(x_original, clip_min, clip_max)
    x_tanh = (x_tanh - clip_min) / (clip_max - clip_min)
    x_tanh = arctanh_approx(((x_tanh * 2) - 1) * tanh_smoother)
    return x_tanh

def arctanh_approx(x):
    return torch.log1p(2*x/(1-x)) / 2

def one_hot_embedding(labels, num_classes):
    """Embedding labels to one-hot form.

    Args:
      labels: (LongTensor) class labels, sized [N,].
      num_classes: (int) number of classes.

    Returns:
      (tensor) encoded labels, sized [N, #classes].
    """
    y = torch.eye(num_classes)
    return y[labels]