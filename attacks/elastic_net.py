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

from attacks.attack_utils import distance_l2_squared, distance_l1, one_hot_embedding, fix_logits, convert_labels, generate_context_attack_indices

ONE_MINUS_EPS = 0.999999
'''We check whether we should abort early every max_iterations // NUM_CHECKS iterations'''
NUM_CHECKS = 10
TARGET_MULT = 10000
REPEAT_STEP = 10
PREV_LOSS_INIT = 1e6
INF = 10e10
UPPER_CHECK = 1e9

class ElasticNetL1Attack():
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
    :param clip_min: mininum value per input dimension.
    :param clip_max: maximum value per input dimension.
    :param beta: hyperparameter trading off L2 minimization for L1 minimization
    :param decision_rule: EN or L1. Select final adversarial example from
                          all successful examples based on the least
                          elastic-net or L1 distortion criterion.
    :param loss_fn: loss function
    """

    def __init__(self, confidence=0,
                 targeted=False, learning_rate=1e-2,
                 binary_search_steps=9, max_iterations=10000,
                 abort_early=False, initial_const=1e-3,
                 clip_min=0., clip_max=1., beta=1e-2, decision_rule='EN',
                 attack_mode='context', class_fraction=1.0, shot_fraction=0.1,
                 success_fraction=0.5, c_upper=10e10, c_lower=0.0):
        """ElasticNet L1 Attack implementation in pytorch."""

        self.confidence = confidence
        self.targeted = targeted
        self.learning_rate = learning_rate
        self.init_learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.binary_search_steps = binary_search_steps
        self.abort_early = abort_early
        self.initial_const = initial_const
        self.beta = beta
        self.clip_range = (clip_min, clip_max)
        # The last iteration (if we run many steps) repeat the search once.
        self.repeat = binary_search_steps >= REPEAT_STEP
        self.decision_rule = decision_rule
        self.attack_mode = attack_mode
        self.class_fraction = class_fraction
        self.shot_fraction = shot_fraction
        self.success_fraction = success_fraction
        self.c_range = (c_lower, c_upper)
        self.global_step = 0

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


    def _is_successful(self, output, target, is_logits):
        # determine success, see if confidence-adjusted logits give the right
        #   label
        if is_logits:
            output = output.detach().clone()
            if self.targeted:
                output[torch.arange(len(target)).long(),
                       target] -= self.confidence
            else:
                output[torch.arange(len(target)).long(),
                       target] += self.confidence
            prediction = torch.argmax(output, dim=1)
        else:
            prediction = output

        if self.targeted:
            return ((prediction == target).sum()/float(prediction.shape[0])).item() >= self.success_fraction
        else:
            return ((prediction != target).sum()/float(prediction.shape[0])).item() >= self.success_fraction

    def _fast_iterative_shrinkage_thresholding(self, x, yy_k, xx_k):
        zt = self.global_step / (self.global_step + 3)

        upper = torch.clamp(yy_k - self.beta, max=self.clip_max)
        lower = torch.clamp(yy_k + self.beta, min=self.clip_min)

        diff = yy_k - x
        cond1 = (diff > self.beta).float()
        cond2 = (torch.abs(diff) <= self.beta).float()
        cond3 = (diff < -self.beta).float()

        xx_k_p_1 = (cond1 * upper) + (cond2 * x) + (cond3 * lower)
        yy_k.data = xx_k_p_1 + (zt * (xx_k_p_1 - xx_k))
        return yy_k, xx_k_p_1

    def generate(self, context_images, context_labels, target_images, model, get_logits_fn, device, target_labels=None):
        # Assert that, if attack is targeted, y is provided:
        if self.targeted and target_labels is None:
            raise ValueError("Target labels `y` need to be provided for a targeted attack.")

        if target_labels is not None:
            # TODO: Is this right?
            import pdb; pdb.set_trace()
            assert len(target_labels.size()) == 1
        else:
            # Generate target labels based on model's initial prediction:
            initial_logits = fix_logits(get_logits_fn(context_images, context_labels, target_images))
            target_labels = convert_labels(initial_logits)

        classes = torch.unique(context_labels)
        num_classes = len(classes)
        # Make one-hot encoding for target_labels
        target_labels_oh = one_hot_embedding(target_labels, num_classes).to(device)
        adv_context_indices = generate_context_attack_indices(context_labels, self.class_fraction, self.shot_fraction)

        #I don't think this is necessary:
        #x = x.detach().clone()
        # We optimize the entire context set as a single attack, so we only have one c
        c_current = self.c_range[0]
        c_lower_bound = 0.0
        c_upper_bound = self.c_range[1]

        # As many best_dists as we have adversarial context images
        o_best_dist = torch.ones(len(adv_context_indices), device=device) * INF
        # As many best_labels as we have target images
        o_best_labels = torch.ones(target_images.shape[0], device=device) * -1.0
        # Store entire adversarial context set
        o_best_attack = context_images.clone()

        # Start binary search
        for outer_step in range(self.binary_search_steps):
            #TODO: Change this to something we pass in to all the functions that need it?
            self.global_step = 0

            # slack vector from the paper
            yy_k = nn.Parameter(context_images.clone())
            xx_k = context_images.clone()

            curr_dist = torch.ones(len(adv_context_indices), device=device) * INF
            curr_labels = -torch.ones(target_images.shape[0], device=device)

            previous_loss = PREV_LOSS_INIT

            if self.repeat and outer_step == (self.binary_search_steps - 1):
                c_current = c_upper_bound

            lr = self.learning_rate

            for k in range(self.max_iterations):

                # reset gradient
                if yy_k.grad is not None:
                    yy_k.grad.detach_()
                    yy_k.grad.zero_()

                # loss over yy_k with only L2 same as C&W
                # we don't update L1 loss with SGD because we use ISTA
                target_outputs = fix_logits(get_logits_fn(yy_k, context_labels, target_images))
                l2_dist = distance_l2_squared(yy_k, context_images)
                loss_opt = self._loss_fn(target_outputs, target_labels_oh, None, l2_dist, c_current, exclude_l1=True)
                loss_opt.backward()

                # In-place gradient step, y = y - lr * y.grad
                yy_k.data.add_(-lr, yy_k.grad.data)
                self.global_step += 1

                # polynomial decay of learning rate
                lr = self.init_learning_rate * (1 - self.global_step / self.max_iterations)**0.5

                yy_k, xx_k = self._fast_iterative_shrinkage_thresholding(context_images, yy_k, xx_k)

                # loss ElasticNet or L1 over xx_k
                with torch.no_grad():
                    target_outputs = fix_logits(get_logits_fn(xx_k, context_labels, target_images))
                    l2_dist = distance_l2_squared(xx_k, context_images)
                    l1_dist = distance_l1(xx_k, context_images)

                    if self.decision_rule == 'EN':
                        dist = l2_dist + (l1_dist * self.beta)
                    elif self.decision_rule == 'L1':
                        dist = l1_dist
                    loss = self._loss_fn(target_outputs, target_labels_oh, l1_dist, l2_dist, c_current)

                    if self.abort_early:
                        if k % (self.max_iterations // NUM_CHECKS or 1) == 0:
                            if loss > previous_loss * ONE_MINUS_EPS:
                                break
                            previous_loss = loss

                    _, target_output_labels = torch.max(target_outputs, 1)

                    if self._is_successful(target_outputs, target_labels, True):
                        # If dist is better than best dist for curr c, update
                        if dist.sum() < curr_dist.sum():
                            curr_dist = dist
                            curr_labels = target_output_labels
                        if dist.sum() < o_best_dist.sum():
                            o_best_dist = dist
                            o_best_labels = target_output_labels
                            o_best_attack = xx_k.data.clone()

            # Update c_current for next iteration:
            if self._is_successful(curr_labels, target_labels):
                c_upper_bound = min(c_upper_bound, c_current)

                if c_upper_bound < UPPER_CHECK:
                    c_current = (c_lower_bound + c_upper_bound) / 2.0
            else:
                c_lower_bound = max(c_lower_bound, c_current)
                if c_upper_bound < UPPER_CHECK:
                    c_current = (c_lower_bound + c_upper_bound) / 2.0
                else:
                    c_current *= 10

        return o_best_attack, adv_context_indices
