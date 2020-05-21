from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MAML(nn.Module):
    """Simple MAML network for few-shot image classification. Uses exact
    specification as in original paper. (See https://arxiv.org/abs/1703.03400
    for details).

    Args:
        update_lr (float): inner loop learning rate
        in_channels (int): number of input image channels
        num_classes (int): "way" of classification task
        gradient_steps (int): number of inner loop training gradient steps
        (Default: 1)
        num_channels (int): number of hidden channels in conv layers
        (Default: 64)
        kernel_width (int): width of convolutional kernel
        (Default: 3)
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
        super(MAML, self).__init__()
        self.update_lr = update_lr
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.gradient_steps = gradient_steps
        self.num_channels = num_channels
        self.kernel_size = kernel_width
        self.create_graph = not first_order
        self.max_pool = max_pool
        self.flatten = flatten
        self.hidden_dim = hidden_dim

        # Some harcoded values
        self.relu = nn.ReLU()

        # Initialize BN layers
        self.bn1 = nn.BatchNorm2d(self.num_channels)
        self.bn2 = nn.BatchNorm2d(self.num_channels)
        self.bn3 = nn.BatchNorm2d(self.num_channels)
        self.bn4 = nn.BatchNorm2d(self.num_channels)
        
        # Construct model parameters
        self.weights = self.construct_weights()

    def forward(self, x, weights=None):
        """Forward pass through the model

        Args:
             x (torch.tensor): inputs to model
             (N x C x H x W)
             weights (dict::params): dictionary containing model parameters
             (Default: None, use global parameters)
        Returns
            (torch.tensor) model outputs
            (N x num_classes)
        """
        if not weights:
            weights = self.weights

        h = self.feature_extractor(x, weights)
        return F.linear(h, weights['w5'], weights['b5'])

    def feature_extractor(self, x, weights):
        """Forward pass through model feature extractor (before linear
        classifier.
        
        Args:
             x (torch.tensor): inputs to model
             (N x C x H x W)
             weights (dict::params): dictionary containing model parameters
             (Default: None, use global parameters)
        Returns
            (torch.tensor) model outputs
            (N x hidden_dim)
        """
        h = self.conv_block(x, weights['conv1'], weights['b1'], self.bn1)
        h = self.conv_block(h, weights['conv2'], weights['b2'], self.bn2)
        h = self.conv_block(h, weights['conv3'], weights['b3'], self.bn3)
        h = self.conv_block(h, weights['conv4'], weights['b4'], self.bn4)
        if self.flatten:
            h = h.view(h.size(0), -1)
        else:
            h = torch.mean(h, dim=(2,3), keepdim=False)  
        return h

    def compute_objective(self,
                          x_context,
                          y_context,
                          x_target,
                          y_target,
                          accuracy=False):
        """Compute the MAML objective function, consisting of inner loop
        to update local parameters, and return loss on target set with updated
        model parameters.

        Args:
            x_context (torch.tensor): Context inputs
            (N_c x C x H x W)
            y_context (torch.tensor): Context outputs
            (N_c x num_classes)
            x_target (torch.tensor): Target inputs
            (N_t x C x H x W)
            y_target (torch.tensor): Target outputs
            (N_t x num_classes)
            accuracy (bool): If True, compute and return accuracy as well as
            loss. (Default: True)
        """
        phi_t, losses = self.inner_loop(x_context,
                                        y_context,
                                        self.gradient_steps)
        y_ = self(x_target, phi_t)
        loss = F.cross_entropy(y_, y_target, reduction='sum')
        
        # Compute target set accuracy
        if accuracy:
            correct_preds = torch.eq(y_.argmax(dim=-1), y_target)
            acc = correct_preds.type(torch.float).mean()
            return loss, acc
        else:
            return loss

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
        y_ = self(x_context, self.weights)
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
            y_ = self(x_context, fast_weights)
            loss = F.cross_entropy(y_, y_context)

            # Compute gradients and update weights
            gradients = self.compute_gradients(loss, fast_weights)
            fast_weights = gradient_step(fast_weights, gradients)
            losses.append(loss.data)

        return fast_weights, losses

    def construct_weights(self):
        """Initialize global parameters of model"""
        weights = nn.ParameterDict()

        def init_layer(shape):
            """Helper function to initialize Conv2d layer parameters"""
            w = nn.Parameter(torch.ones(*shape))
            torch.nn.init.xavier_normal_(w)
            b = nn.Parameter(torch.zeros(shape[0]))
            return nn.Parameter(w), nn.Parameter(b)

        # Determine layer shapes
        shape0 = [self.num_channels, self.in_channels, self.kernel_size,
                  self.kernel_size]
        shapel = [self.num_channels, self.num_channels, self.kernel_size,
                  self.kernel_size]
        dense_shape = [self.num_classes, self.hidden_dim]

        # Initialize weights for the shapes
        weights['conv1'], weights['b1'] = init_layer(shape0)
        weights['conv2'], weights['b2'] = init_layer(shapel)
        weights['conv3'], weights['b3'] = init_layer(shapel)
        weights['conv4'], weights['b4'] = init_layer(shapel)
        weights['w5'], weights['b5'] = init_layer(dense_shape)

        return weights

    def conv_block(self, x, w, b, bn):
        """Forward pass through a convolutional block"""
        # Apply convolution with max-pooling / down-sampling
        if self.max_pool:
            h = F.conv2d(input=x, weight=w, bias=b, padding=1)
        else:
            h = F.conv2d(input=x, weight=w, bias=b, stride=2, padding=1)
        # Conv ReLU BN (PyTorch codebase)
        h = bn(self.relu(h))
        # Conv BN ReLU (Original MAML)
        # h = self.relu(bn(h))
        if self.max_pool:
            h = F.max_pool2d(h, kernel_size=2, stride=2)

        return h

    def compute_gradients(self, loss, inputs):
        """Construct an ordered dictionary with gradients for a given loss

        Args:
            loss (torch.scalar): loss to take derivative of
            inputs (dict::tensors): tensors which require gradients

        Returns:
            (dict::tensors): dictionary of gradients with same structure as
            values
        """
        return OrderedDict(zip(
            inputs.keys(),
            torch.autograd.grad(loss, inputs.values(),
                                create_graph=self.create_graph)
        ))

    def set_gradient_steps(self, gradient_steps):
        """Change default number of gradient steps used in forward pass"""
        self.gradient_steps = gradient_steps

