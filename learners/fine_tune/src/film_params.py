import torch
import torch.nn as nn


class FilmAdapter(nn.Module):
    def __init__(self, layer, num_maps, num_blocks):
        super().__init__()
        self.num_maps = num_maps
        self.num_blocks = num_blocks
        self.num_target_layers = len(self.num_maps)
        self.layer = layer
        self.layers = self.get_layers()

    def get_layers(self):
        layers = nn.ModuleList()
        for num_maps, num_blocks in zip(self.num_maps, self.num_blocks):
            layers.append(
                self.layer(
                    num_maps=num_maps,
                    num_blocks=num_blocks,
                )
            )
        return layers

    def forward(self, x):
        return [self.layers[layer](x) for layer in range(self.num_target_layers)]

    def regularization_term(self):
        l2_term = 0
        for layer in self.layers:
            l2_term += layer.regularization_term()
        return l2_term


class FilmLayer(nn.Module):
    def __init__(self, num_maps, num_blocks):
        super(FilmLayer, self).__init__()

        self.num_maps = num_maps
        self.num_blocks = num_blocks

        self.gamma = nn.ParameterList()
        self.beta = nn.ParameterList()
        self.gamma_regularizers = nn.ParameterList()
        self.beta_regularizers = nn.ParameterList()

        for i in range(self.num_blocks):
            self.gamma.append(nn.Parameter(torch.ones(self.num_maps[i]), requires_grad=True))
            self.beta.append(nn.Parameter(torch.zeros(self.num_maps[i]), requires_grad=True))
            self.gamma_regularizers.append(nn.Parameter(nn.init.normal_(torch.empty(num_maps[i]), 0, 0.001),
                                                        requires_grad=True))
            self.beta_regularizers.append(nn.Parameter(nn.init.normal_(torch.empty(num_maps[i]), 0, 0.001),
                                                       requires_grad=True))

    def forward(self, x):
        block_params = []
        for block in range(self.num_blocks):
            block_param_dict = {
                'gamma': self.gamma[block] * self.gamma_regularizers[block] +
                         torch.ones_like(self.gamma_regularizers[block]),
                'beta': self.beta[block] * self.beta_regularizers[block]
            }
            block_params.append(block_param_dict)
        return block_params

    def regularization_term(self):
        """
        Compute the regularization term for the parameters. Recall, FiLM applies gamma * x + beta. As such, params
        gamma and beta are regularized to unity, i.e. ||gamma - 1||_2 and ||beta||_2.
        :return: (torch.tensor) Scalar for l2 norm for all parameters according to regularization scheme.
        """
        l2_term = 0
        for gamma_regularizer, beta_regularizer in zip(self.gamma_regularizers, self.beta_regularizers):
            l2_term += (gamma_regularizer ** 2).sum()
            l2_term += (beta_regularizer ** 2).sum()
        return l2_term


class NullFeatureAdaptationNetwork(nn.Module):
    """
    Dummy adaptation network for the case of "no_adaptation".
    """
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return {}

    @staticmethod
    def regularization_term():
        return 0
