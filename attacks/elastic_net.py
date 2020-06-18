# Copyright (c) 2019-present, Alexandre Araujo.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
import torch.nn as nn

from attacks.attack_utils import distance_l2_squared, distance_l1, one_hot_embedding, fix_logits, convert_labels, \
    generate_context_attack_indices, Logger

ONE_MINUS_EPS = 0.999999
'''We check whether we should abort early every max_iterations // NUM_CHECKS iterations'''
NUM_CHECKS = 10
TARGET_MULT = 10000
REPEAT_STEP = 10
PREV_LOSS_INIT = 1e6
INF = 10e10
UPPER_CHECK = 1e9

class ElasticNet():
    """
    The ElasticNet L1 Attack, https://arxiv.org/abs/1709.04114

    :param predict: forward pass function.
    :param num_classes: number of clasess.
    :param confidence: confidence of the adversarial examples.
    :param targeted: if the attack is targeted.
    :param learning_rate: the learning rate for the attack algorithm
    :param binary_search_steps: number of binary search times to find the
        optimum
    :param max_iterations: the maximum number of iterations
    :param abort_early: if set to true, abort early if getting stuck in local
        min
    :param initial_const: initial value of the constant c
    :param beta: hyperparameter trading off L2 minimization for L1 minimization
    :param decision_rule: EN or L1. Select final adversarial example from
                          all successful examples based on the least
                          elastic-net or L1 distortion criterion.
    :param loss_fn: loss function
    """

    def __init__(self,
                 checkpoint_dir,
                 confidence=0,
                 targeted=False, learning_rate=1e-2,
                 binary_search_steps=9, max_iterations=1000,
                 abort_early=False, beta=1e-2, decision_rule='EN',
                 attack_mode='context', class_fraction=1.0, shot_fraction=0.1,
                 success_fraction=0.5, c_upper=10e10, c_lower=1e-3):
        """ElasticNet L1 Attack implementation in pytorch."""

        self.confidence = confidence
        self.targeted = targeted
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.binary_search_steps = binary_search_steps
        self.abort_early = abort_early
        self.beta = beta
        # The last iteration (if we run many steps) repeat the search once.
        self.decision_rule = decision_rule
        self.attack_mode = attack_mode
        self.class_fraction = class_fraction
        self.shot_fraction = shot_fraction
        self.success_fraction = success_fraction
        self.c_range = (c_lower, c_upper)

        self.init_learning_rate = learning_rate
        self.global_step = 0
        self.repeat = binary_search_steps >= REPEAT_STEP
        # Set defaults, but we'll use the context set to calculate these when generating attacks
        self.logger = Logger(checkpoint_dir, "enet_logs.txt")

    def get_attack_mode(self):
        return self.attack_mode

    def _loss_fn(self, target_outputs, target_labels_oh, l1_dist, l2_dist_squared, const, exclude_l1=False):
        real = (target_labels_oh * target_outputs).sum(dim=1)
        other = ((1.0 - target_labels_oh) * target_outputs -
                 (target_labels_oh * TARGET_MULT)).max(1)[0]

        if self.targeted:
            loss_logits = torch.clamp(other - real + self.confidence, min=0.)
        else:
            loss_logits = torch.clamp(real - other + self.confidence, min=0.)
        loss_logits = torch.sum(const * loss_logits)

        loss_l2 = l2_dist_squared.sum()

        if exclude_l1:
            loss = loss_logits + loss_l2
        else:
            loss_l1 = self.beta * l1_dist.sum()
            loss = loss_logits + loss_l2 + loss_l1
        return loss

    def _context_attack_successful(self, output, target, is_logits=False):
        # determine success, see if confidence-adjusted logits give the right label
        if is_logits:
            output = output.detach()
            if self.targeted:
                output[torch.arange(len(target)).long(),
                       target] -= self.confidence
            else:
                output[torch.arange(len(target)).long(),
                       target] += self.confidence
            prediction = torch.argmax(output, dim=1)
        else:
            # Labels are all -1 and thus invalid. Attack not successful
            if (output == -1).sum() == output.shape[0]:
                return False
            else:
                prediction = output.long()

        if self.targeted:
            return ((prediction == target).sum()/float(prediction.shape[0])).item() >= self.success_fraction
        else:
            return ((prediction != target).sum()/float(prediction.shape[0])).item() >= self.success_fraction

    def _target_attack_successful(self, output, target, is_logits=False):
        # determine success, see if confidence-adjusted logits give the right label
        if is_logits:
            output = output.detach().clone()
            if self.targeted:
                output[target] -= self.confidence
            else:
                output[target] += self.confidence
            prediction = torch.argmax(output)
        else:
            # Labels are all -1 and thus invalid. Attack not successful
            if output == -1:
                return False
            else:
                prediction = output.long()

        if self.targeted:
            return prediction == target
        else:
            return prediction != target

    def _fast_iterative_shrinkage_thresholding(self, x, yy_k, xx_k, clip_min, clip_max):
        zt = self.global_step / (self.global_step + 3)

        upper = torch.clamp(yy_k - self.beta, max=clip_max)
        lower = torch.clamp(yy_k + self.beta, min=clip_min)
        # Zero for all non-adv context images
        diff = yy_k - x
        cond1 = (diff > self.beta).float()
        cond2 = (torch.abs(diff) <= self.beta).float()
        cond3 = (diff < -self.beta).float()

        xx_k_p_1 = (cond1 * upper) + (cond2 * x) + (cond3 * lower)
        yy_k.data = xx_k_p_1 + (zt * (xx_k_p_1 - xx_k))
        return yy_k, xx_k_p_1

    def generate(self, context_images, context_labels, target_images, model, get_logits_fn, device, target_labels=None):
        # Assert that, if attack is targeted, y is provided:

        self.logger.print_and_log(
            "Performing Elastic Net attack on {} set. Settings = (confidence={}, beta={}, decision_rule={}, "
            "binary_search_steps={}, max_iterations={}, abort_early={}, learning_rate={}, "
            "success_fraction={})".format(self.attack_mode, self.confidence, self.beta, self.decision_rule,
                                          self.binary_search_steps, self.max_iterations, self.abort_early,
                                          self.learning_rate, self.success_fraction))
        self.logger.print_and_log(
            "class_fraction = {}, shot_fraction = {}".format(self.class_fraction, self.shot_fraction))

        if self.targeted and target_labels is None:
            raise ValueError("Target labels `y` need to be provided for a targeted attack.")

        if target_labels is not None:
            assert len(target_labels.size()) == 1
        else:
            # Generate target labels based on model's initial prediction:
            initial_logits = fix_logits(get_logits_fn(context_images, context_labels, target_images))
            target_labels = convert_labels(initial_logits)

        classes = torch.unique(context_labels)
        num_classes = len(classes)
        # Make one-hot encoding for target_labels
        target_labels_oh = one_hot_embedding(target_labels, num_classes).to(device)

        if self.attack_mode == 'context':
            # These indices are tensors. I'm not sure that's what we want, but I don't want to change it without further discussion
            adv_context_indices_t = generate_context_attack_indices(context_labels, self.class_fraction, self.shot_fraction)
            adv_context_indices = [index_tensor.item() for index_tensor in adv_context_indices_t]
            attack_set = context_images
            # We are conceptually only performing one attack, albeit over a set of images
            num_attacks = 1
            num_inputs = len(adv_context_indices)
        else:
            attack_set = target_images
            num_attacks = len(target_images)
            num_inputs = num_attacks

        clip_max = attack_set.max().item()
        clip_min = attack_set.min().item()

        # I don't think this is necessary:
        # x = x.detach().clone()

        # For context set attacks, we optimize the entire context set as a single attack, so we only have one c
        c_current = torch.ones(num_attacks, device=device) * self.c_range[0]
        c_lower_bound = torch.zeros(num_attacks, device=device)
        c_upper_bound = torch.ones(num_attacks, device=device) * self.c_range[1]

        # As many best_dists as we have adversarial context images
        o_best_dist = torch.ones(num_inputs, device=device) * INF
        # As many best_labels as we have target images
        # o_best_labels = torch.ones(target_images.shape[0], device=device) * -1.0
        # Store entire adversarial context set
        o_best_attack = attack_set.clone()

        # Start binary search
        for outer_step in range(self.binary_search_steps):
            #TODO: Change this to something we pass in to all the functions that need it?
            self.global_step = 0
            self.logger.print_and_log('Using scale const: {}'.format(c_current))

            # slack vector from the paper
            yy_k = attack_set.clone()
            xx_k = attack_set.clone()

            curr_dist = torch.ones(num_inputs, device=device) * INF
            curr_labels = -torch.ones(target_images.shape[0], device=device) #TODO: check this

            previous_loss = PREV_LOSS_INIT

            if self.repeat and outer_step == (self.binary_search_steps - 1):
                c_current = c_upper_bound

            lr = self.learning_rate

            for k in range(self.max_iterations):
                model.zero_grad()
                yy_k.requires_grad = True

                # loss over yy_k with only L2 same as C&W
                # we don't update L1 loss with SGD because we use ISTA
                if self.attack_mode == 'context':
                    target_outputs = fix_logits(get_logits_fn(yy_k, context_labels, target_images))
                    l2_dist = distance_l2_squared(yy_k[adv_context_indices], context_images[adv_context_indices])
                else:
                    target_outputs = fix_logits(get_logits_fn(context_images, context_labels, yy_k))
                    l2_dist = distance_l2_squared(yy_k, target_images)
                # TODO: Are we doing any sort of weighting for the l2_dists to compensate for input_num?
                loss_opt = self._loss_fn(target_outputs, target_labels_oh, None, l2_dist, c_current, exclude_l1=True)
                loss_opt.backward()

                yy_k_grad = yy_k.grad
                yy_k = yy_k.detach()
                # Gradient step
                if self.attack_mode == 'context':
                    for i, index in enumerate(adv_context_indices):
                        yy_k[index] = yy_k[index] - lr* yy_k_grad[index]
                else:
                    yy_k = yy_k - lr * yy_k_grad

                self.global_step += 1

                # polynomial decay of learning rate
                lr = self.init_learning_rate * (1 - self.global_step / self.max_iterations)**0.5

                if self.attack_mode == 'context':
                    yy_k_new, xx_k_new = self._fast_iterative_shrinkage_thresholding(context_images[adv_context_indices], yy_k[adv_context_indices], xx_k[adv_context_indices], clip_min, clip_max)

                    for (i, index) in enumerate(adv_context_indices):
                        yy_k[index] = yy_k_new[i]
                        xx_k[index] = xx_k_new[i]
                else:
                    yy_k, xx_k = self._fast_iterative_shrinkage_thresholding(target_images, yy_k, xx_k, clip_min, clip_max)

                # loss ElasticNet or L1 over xx_k
                with torch.no_grad():
                    if self.attack_mode == 'context':
                        target_outputs = fix_logits(get_logits_fn(xx_k, context_labels, target_images))
                        l2_dist = distance_l2_squared(xx_k[adv_context_indices], context_images[adv_context_indices])
                        l1_dist = distance_l1(xx_k[adv_context_indices], context_images[adv_context_indices])
                    else:
                        target_outputs = fix_logits(get_logits_fn(context_images, context_labels, xx_k))
                        l2_dist = distance_l2_squared(xx_k, target_images)
                        l1_dist = distance_l1(xx_k, target_images)

                    if self.decision_rule == 'EN':
                        dist = l2_dist + (l1_dist * self.beta)
                    elif self.decision_rule == 'L1':
                        dist = l1_dist
                    loss = self._loss_fn(target_outputs, target_labels_oh, l1_dist, l2_dist, c_current)

                    if k % 10 == 0:
                        self.logger.print_and_log('optim step [{}] loss: {}'.format(k, loss))

                    if self.abort_early:
                        if k % (self.max_iterations // NUM_CHECKS or 1) == 0:
                            if loss > previous_loss * ONE_MINUS_EPS:
                                break
                            previous_loss = loss

                    _, target_output_labels = torch.max(target_outputs, 1)

                    if self.attack_mode == 'context':
                        if self._context_attack_successful(target_outputs, target_labels, is_logits=True):
                            # If dist is better than best dist for curr c, update
                            if dist.sum() < curr_dist.sum():
                                curr_dist = dist
                                curr_labels = target_output_labels
                            if dist.sum() < o_best_dist.sum():
                                o_best_dist = dist
                                o_best_attack = xx_k.data.clone()
                    else:
                        for i in range(num_attacks):
                            if self._target_attack_successful(target_outputs[i], target_labels[i], is_logits=True):
                                # If dist is better than best dist for curr c, update
                                if dist.sum[i] < curr_dist[i]:
                                    curr_dist[i] = dist[i]
                                    curr_labels[i] = target_output_labels[i]
                                if dist[i] < o_best_dist[i]:
                                    o_best_dist[i] = dist[i]
                                    o_best_attack[i] = xx_k.data.clone()

            # Update c_current for next iteration:
            for i in range(num_attacks):
                if (self.attack_mode == 'context' and self._context_attack_successful(curr_labels, target_labels, is_logits=False)) \
                        or (self.attack_mode == 'target' and self._target_attack_successful(curr_labels[i], target_labels[i], is_logits=False)):
                    self.logger.print_and_log("Found successful attack")
                    c_upper_bound[i] = min(c_upper_bound[i], c_current[i])

                    if c_upper_bound[i] < UPPER_CHECK:
                        c_current[i] = (c_lower_bound[i] + c_upper_bound[i]) / 2.0
                else:
                    self.logger.print_and_log("Not successful attack")
                    c_lower_bound[i] = max(c_lower_bound[i], c_current[i])
                    if c_upper_bound[i] < UPPER_CHECK:
                        c_current[i] = (c_lower_bound[i] + c_upper_bound[i]) / 2.0
                    else:
                        c_current[i] *= 10

        return o_best_attack, adv_context_indices
