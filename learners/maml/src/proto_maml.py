from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from learners.maml.src.maml import MAML

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ProtoMAML(MAML):
    """
    New version of proto-MAML: complete MAML, but instead of learning
    initialization for classifier weights, these are initialized to class mean
    representations, as in proto-nets.
    """
    def __init__(self,
                 update_lr,
                 in_channels,
                 num_classes,
                 gradient_steps=1,
                 num_channels=32,
                 kernel_width=3,
                 first_order=True,
                 max_pool=True,
                 flatten=True,
                 hidden_dim=800):
        super(ProtoMAML, self).__init__(update_lr,
                                        in_channels,
                                        num_classes,
                                        gradient_steps,
                                        num_channels,
                                        kernel_width,
                                        first_order,
                                        max_pool,
                                        flatten,
                                        hidden_dim)

    def inner_loop(self, x_context, y_context, num_steps):
        """Run inner loop to get task specific weights. Final classification
        weights / biases are initialized from class means.

        Args:
             x_context (torch.tensor): Context set inputs
             (N x C x H x W)
             y_context (torch.tensor): Context set outputs
             (N x num_classes)
             num_steps (int): Number of gradient steps for optimization

        Returns:
              (dict::params): weight dictionary containing phi_t
        """
        # Complete initialization for task
        class_reps = self.feature_extractor(x_context, self.weights)
        w, b = self._get_task_classifier(class_reps, y_context)
        self.weights['w5'] = nn.Parameter(w)
        self.weights['b5'] = nn.Parameter(b)

        # Run inner loop as usual
        return super(ProtoMAML, self).inner_loop(x_context, y_context, 
                                                 num_steps)

    def construct_weights(self):
        """
        Initialize global parameters of model. Here we do not use final
        layer weights initializer
        """
        weights = super(ProtoMAML, self).construct_weights()
        del weights['w5']
        del weights['b5']
        return weights

    def _get_task_classifier(self, context_reps, y_context):
        """Construct initial classifier weights from mean representations"""
        # Preamble to classifier initialization
        num_classes = torch.max(y_context) + 1
        num_classes = num_classes.item()
        weights = torch.empty(num_classes, self.hidden_dim).to(device)
        biases = torch.empty(num_classes).to(device)
        
        # For each class, compute weights and biases initialization
        for c in range(num_classes):
            class_indices = torch.where(y_context == c)[0]
            class_rep = context_reps[class_indices].mean(0)
            weights[c] = class_rep
            biases[c] = -0.5 * class_rep @ class_rep.t()

        return weights, biases


class ProtoMAMLv2(ProtoMAML):
    """
        New version of proto-MAML: complete MAML, but instead of learning
        initialization for classifier weights, these are initialized to class mean
        representations, as in proto-nets.
        """

    def __init__(self,
                 update_lr,
                 in_channels,
                 num_classes,
                 gradient_steps=1,
                 num_channels=32,
                 kernel_width=3,
                 first_order=True,
                 max_pool=True,
                 flatten=True,
                 hidden_dim=800):
        super(ProtoMAMLv2, self).__init__(update_lr,
                                          in_channels,
                                          num_classes,
                                          gradient_steps,
                                          num_channels,
                                          kernel_width,
                                          first_order,
                                          max_pool,
                                          flatten,
                                          hidden_dim)

    def inner_loop(self, x_context, y_context, num_steps):
        """Run inner loop to get task specific weights

        Args:
             x_context (torch.tensor): Context set inputs
             (N x C x H x W)
             y_context (torch.tensor): Context set outputs
             (N x num_classes)
             num_steps (int): Number of gradient steps for optimization

        Returns:
              (dict::params): weight dictionary containing phi_t
        """
        # Preamble to inner loop
        losses = []

        # First step taken with initialized weights
        y_ = self._inner_loop_forward(x_context, y_context, self.weights)
        loss = F.cross_entropy(y_, y_context)
        losses.append(loss.data)

        # Helper function to update weights with gradients
        def gradient_step(ws, gvs):
            return OrderedDict(zip(
                ws.keys(),
                [ws[key] - self.update_lr * gvs[key]
                 for key in ws.keys()]
            ))

        # Compute gradients and initialize fast weights
        gradients = self.compute_gradients(loss, self.weights)
        fast_weights = gradient_step(self.weights, gradients)

        # Enter inner gradient loop
        for step in range(num_steps - 1):
            # Forward pass with current weights and compute loss
            y_ = self._inner_loop_forward(x_context, y_context, fast_weights)
            loss = F.cross_entropy(y_, y_context)

            # Compute gradients and update weights
            gradients = self.compute_gradients(loss, fast_weights)
            fast_weights = gradient_step(fast_weights, gradients)
            losses.append(loss.data)

        # Compute final version of classifier weights
        context_reps = self.feature_extractor(x_context, fast_weights)
        fast_weights['w5'], fast_weights['b5'] = \
            self._get_task_classifier(context_reps, y_context)
        return fast_weights, losses

    def _inner_loop_forward(self, x_context, y_context, weights=None):
        """Inner loop forward in a ProtoNets fashion"""
        if not weights:
            weights = self.weights

        h = self.feature_extractor(x_context, weights)
        w, b = self._get_task_classifier(h, y_context)

        return F.linear(h, w, b)

