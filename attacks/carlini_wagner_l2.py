"""
Carlini-Wagner attack (http://arxiv.org/abs/1608.04644).

Implementation based on: https://github.com/kkew3/pytorch-cw2/blob/master/cw.py

Referential implementation:

- https://github.com/carlini/nn_robust_attacks.git (the original implementation)
- https://github.com/rwightman/pytorch-nips2017-attack-example.git
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from attacks.attack_utils import convert_labels, generate_context_attack_indices, fix_logits, one_hot_embedding




def atanh(x, eps=1e-6):
    """
    The inverse hyperbolic tangent function, missing in pytorch.

    :param x: a Tensor
    :param eps: used to enhance numeric stability
    :return: :math:`\\tanh^{-1}{x}`, of the same type as ``x``
    """
    x = x * (1 - eps)
    return 0.5 * torch.log((1.0 + x) / (1.0 - x))


def to_tanh_space(x, box):
    """
    Convert a batch of tensors to tanh-space. This method complements the
    implementation of the change-of-variable trick in terms of tanh.

    :param x: the batch of tensors, of dimension [B x C x H x W]
    :param box: a tuple of lower bound and upper bound of the box constraint
    :return: the batch of tensors in tanh-space, of the same dimension;
             the returned tensor is on the same device as ``x``
    """
    _box_mul = (box[1] - box[0]) * 0.5
    _box_plus = (box[1] + box[0]) * 0.5
    return atanh((x - _box_plus) / _box_mul)


def from_tanh_space(x, box):
    """
    Convert a batch of tensors from tanh-space to oridinary image space.
    This method complements the implementation of the change-of-variable trick
    in terms of tanh.

    :param x: the batch of tensors, of dimension [B x C x H x W]
    :param box: a tuple of lower bound and upper bound of the box constraint
    :return: the batch of tensors in ordinary image space, of the same
             dimension; the returned tensor is on the same device as ``x``
    """
    _box_mul = (box[1] - box[0]) * 0.5
    _box_plus = (box[1] + box[0]) * 0.5
    return torch.tanh(x) * _box_mul + _box_plus


class CarliniWagnerL2(object):
    """
    The L2 attack adversary. To enforce the box constraint, the
    change-of-variable trick using tanh-space is adopted.

    The loss function to optimize:

    .. math::
        \\|\\delta\\|_2^2 + c \\cdot f(x + \\delta)

    where :math:`f` is defined as

    .. math::
        f(x') = \\max\\{0, (\\max_{i \\ne t}{Z(x')_i} - Z(x')_t) \\cdot \\tau + \\kappa\\}

    where :math:`\\tau` is :math:`+1` if the adversary performs targeted attack;
    otherwise it's :math:`-1`.

    Usage::
    #TODO:
        attacker = L2Adversary()
        # inputs: a batch of input tensors
        # targets: a batch of attack targets
        # model: the model to attack
        advx = attacker(model, inputs, targets)


    The change-of-variable trick
    ++++++++++++++++++++++++++++

    Let :math:`a` be a proper affine transformation.

    1. Given input :math:`x` in image space, map :math:`x` to "tanh-space" by

    .. math:: \\hat{x} = \\tanh^{-1}(a^{-1}(x))

    2. Optimize an adversarial perturbation :math:`m` without constraint in the
    "tanh-space", yielding an adversarial example :math:`w = \\hat{x} + m`; and

    3. Map :math:`w` back to the same image space as the one where :math:`x`
    resides:

    .. math::
        x' = a(\\tanh(w))

    where :math:`x'` is the adversarial example, and :math:`\\delta = x' - x`
    is the adversarial perturbation.

    Since the composition of affine transformation and hyperbolic tangent is
    strictly monotonic, $\\delta = 0$ if and only if $m = 0$.

    Symbols used in docstring
    +++++++++++++++++++++++++

    - ``B``: the batch size
    - ``C``: the number of channels
    - ``H``: the height
    - ``W``: the width
    - ``M``: the number of classes
    """

    def __init__(self, targeted=False,
                 confidence=0.0,
                 c_lower=1e-3,
                 c_upper=1e10,
                 binary_search_steps=10,
                 max_iterations=100,
                 abort_early=True,
                 box_lower=-1.,
                 box_upper=1.,
                 optimizer_lr=1e-2,
                 init_rand=False,
                 attack_mode='context',
                 class_fraction=0.5,
                 shot_fraction=0.5,
                 success_fraction=0.5
                 ):
        """
        :param targeted: ``True`` to perform targeted attack in ``self.run``
               method
        :type targeted: bool
        :param confidence: the confidence constant, i.e. the $\\kappa$ in paper
        :type confidence: float
        :param c_lower: lower bound for the search range of the constant :math:`c`
               Note that we start searching for c at ``lower_bound'' and work our
               way upward as necessary to larger c
        :type c_lower: float
        :param c_upper: upper bound for the search range of the constant :math:`c`
               Note that we start searching for c at ``lower_bound'' and work our
               way upward as necessary to larger c
        :type c_lower: float
        :param binary_search_steps: the number of steps to perform binary search of
               the constant :math:`c` over c_range=[``c_lower``, ``c_upper'']
        :type binary_search_steps: int
        :param max_iterations: the maximum number of optimization steps for each
               constant :math:`c`
        :type max_iterations: int
        :param abort_early: ``True`` to abort early in process of searching for
               :math:`c` when the loss virtually stops increasing
        :type abort_early: bool
        :param box_lower: lower bound of the box in which we do optimization
        :type box_lower: float
        :param box_upper: upper bound of the box in which we do optimization
        :type box_upper: float
        :param optimizer_lr: the base learning rate of the Adam optimizer used
               over the adversarial perturbation in clipped space
        :type optimizer_lr: float
        :param init_rand: ``True`` to initialize perturbation to small Gaussian;
               False is consistent with the original paper, where the
               perturbation is initialized to zero
        :type init_rand: bool
        :param attack_mode: can be either ``context'' to attack the context set or
                ``target'' to attack the target set
        :type attack_mode: string
        :param class_fraction: fraction of classes to attack
        :type class_fraction: float
        :param shot_fraction: fraction of patterns to attack (per class)
        :type shot_fraction: float

        Why to make ``box`` default to (-1., 1.) rather than (0., 1.)? TL;DR the
        domain of the problem in pytorch is [-1, 1] instead of [0, 1].
        According to Xiang Xu (samxucmu@gmail.com)::

        > The reason is that in pytorch a transformation is applied first
        > before getting the input from the data loader. So image in range [0,1]
        > will subtract some mean and divide by std. The normalized input image
        > will now be in range [-1,1]. For this implementation, clipping is
        > actually performed on the image after normalization, not on the
        > original image.

        Why to ``optimizer_lr`` default to 1e-2? The optimizer used in Carlini's
        code adopts 1e-2. In another pytorch implementation
        (https://github.com/rwightman/pytorch-nips2017-attack-example.git),
        though, the learning rate is set to 5e-4.
        """
        if c_lower >= c_upper:
            raise ValueError('c_range lower bound ({}) is expected to be less '
                             'than c_range upper bound ({})'.format(c_lower, c_upper))
        if box_lower >= box_upper:
            raise ValueError('box lower bound ({}) is expected to be less than '
                             'box upper bound ({})'.format(box_lower, box_upper))
        self.targeted = targeted
        self.confidence = confidence
        self.c_range = (c_lower, c_upper)
        self.binary_search_steps = binary_search_steps
        self.max_iterations = max_iterations
        self.abort_early = abort_early
        self.ae_tol = 1e-4  # tolerance of early abort
        self.box = (box_lower, box_upper)
        self.optimizer_lr = optimizer_lr

        # `self.init_rand` is not in Carlini's code, it's an attempt in the
        # referencing pytorch implementation to improve the quality of attacks.
        self.init_rand = init_rand

        self.attack_mode = attack_mode
        self.class_fraction = class_fraction
        self.shot_fraction = shot_fraction
        self.success_fraction = success_fraction

        # Since the larger the `scale_const` is, the more likely a successful
        # attack can be found, `self.repeat` guarantees at least attempt the
        # largest scale_const once. Moreover, since the optimal criterion is the
        # L2 norm of the attack, and the larger `scale_const` is, the larger
        # the L2 norm is, thus less optimal, the last attempt at the largest
        # `scale_const` won't ruin the optimum ever found.
        self.repeat = (self.binary_search_steps >= 10)

    def generate(self, context_images, context_labels, target_images, model, get_logits_fn, device, target_labels=None):
        """
        Produce adversarial examples for ``inputs``.
        """
        assert len(context_images.size()) == 4
        assert len(target_images.size()) == 4
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

        # Which context_images to attack:
        adv_context_indices = generate_context_attack_indices(context_labels, self.class_fraction, self.shot_fraction)
        # The number of patterns for which we're generating the attack
        input_size = len(adv_context_indices)  # type: int

        # Set up bounds and initial values for c
        scale_const = self.c_range[0]
        lower_bound = 0.0
        upper_bound = self.c_range[1]

        # Set up holders for the optimal attacks and related info
        # The three "placeholders" are defined as:
        # - `o_best_l2`: The lowest L2 distance between the input and adversarial images
        # - `o_best_advx`: the best performing adversarial context set
        o_best_l2 = torch.ones(input_size, device=device) * 1e4  # placeholder for inf
        o_best_advx = context_images.clone()

        # Necessary conversions of the inputs into tanh space
        # convert `inputs` to tanh-space
        context_images_tanh = self._to_tanh_space(context_images)  # type: torch.FloatTensor

        # `pert_tanh` is essentially the adversarial perturbation in tanh-space.
        # In Carlini's code it's denoted as `modifier`
        pert_tanh = torch.zeros((input_size, context_images.shape[1], context_images.shape[2], context_images.shape[3]),
                                device=device)  # type: torch.FloatTensor
        if self.init_rand:
            nn.init.normal(pert_tanh, mean=0, std=1e-3)
        pert_tanh.requires_grad = True
        optimizer = optim.Adam([pert_tanh], lr=self.optimizer_lr)

        for sstep in range(self.binary_search_steps):
            if self.repeat and sstep == self.binary_search_steps - 1:
                scale_const = upper_bound
            print('Using scale const:', scale_const)

            # the minimum L2 norms of perturbations found during optimization
            best_l2 = torch.ones(input_size, device=device) * 1e4  # As placeholder for np.inf
            # the perturbed predictions corresponding to `best_l2`, to be used in binary search of `scale_const`
            best_l2_ppred = None

            # previous (summed) batch loss, to be used in early stopping policy
            prev_loss = 1e4  # as placeholder for infinity, type: float

            for optim_step in range(self.max_iterations):
                adv_context_set_tanh = context_images_tanh.clone()
                for i, index in enumerate(adv_context_indices):
                    adv_context_set_tanh[index] = adv_context_set_tanh[index] + pert_tanh[i]

                # Map examples back to image space
                adv_context_set = self._from_tanh_space(adv_context_set_tanh)
                #loss_val is 1-D, pert_norms is [num_context], pert_outputs is [num_target], adv_context_images is [num_context C x W x H]
                loss_val, pert_norms, pert_outputs, adv_context_images = \
                    self._optimize(adv_context_set, context_images, context_labels, target_images,
                                   target_labels_oh, scale_const, get_logits_fn, optimizer)
                if optim_step % 10 == 0: print('optim step [{}] loss: {}'.format(optim_step, loss_val))

                if self.abort_early and not optim_step % (self.max_iterations // 10):
                    if loss_val > prev_loss * (1 - self.ae_tol):
                        break
                    prev_loss = loss_val

                # Outputs for target set, given adversarial context set
                pert_predictions = torch.argmax(pert_outputs, dim=1)
                comp_pert_predictions = torch.argmax(self._compensate_confidence(pert_outputs,target_labels),dim=1)
                # If the attack is successful, see if we've improved the loss
                if self._attack_successful(comp_pert_predictions, target_labels):
                    # TODO: I'm not sure why this would be the case. What exactly does comp_pert do?
                    assert torch.all(comp_pert_predictions.eq(pert_predictions))
                    # If this attack has lower perturbation norm, record it
                    total_pert_norm = pert_norms.sum()
                    if total_pert_norm < best_l2.sum():
                        best_l2_ppred = pert_predictions
                        for i, index in enumerate(adv_context_indices):
                            best_l2[i] = pert_norms[index]
                    if total_pert_norm < o_best_l2.sum():
                        o_successful_attack = True
                        o_best_l2_ppred = pert_predictions
                        for i, index in enumerate(adv_context_indices):
                            o_best_l2[i] = pert_norms[index]
                            o_best_advx[index] = adv_context_images[index].clone()

            # binary search of `scale_const`
            #assert best_l2_ppred is None or self._attack_successful(best_l2_ppred, target_labels, optim_step)
            #assert o_best_l2_ppred is None or self._attack_successful(o_best_l2_ppred, target_labels, optim_step)
            if best_l2_ppred is not None:
                print("Found successful attack")
                # successful; attempt to lower `scale_const` by halving it
                if scale_const < upper_bound:
                    upper_bound = scale_const
                # `upper_bounds_np[i] == c_range[1]` implies no solution
                # found, i.e. upper_bounds_np[i] has never been updated by
                # scale_consts_np[i] until
                # `scale_consts_np[i] > 0.1 * c_range[1]`
                if upper_bound < self.c_range[1] * 0.1:
                    scale_consts = (lower_bound + upper_bound) / 2
            else:
                print("Not successful")
                # failure; multiply `scale_const` by ten if no solution
                # found; otherwise do binary search
                if scale_const > lower_bound:
                    lower_bound = scale_const
                if upper_bound < self.c_range[1] * 0.1:
                    scale_const = (lower_bound + upper_bound) / 2
                else:
                    scale_const *= 10
        return o_best_advx, adv_context_indices

    #    def _optimize(self, model, optimizer, inputs_tanh_var, pert_tanh_var,
    #                 targets_oh_var, c_var):
    # Instead of passing in the tanh context_images and perturbation, pass in the adversarial image and the originals
    def _optimize(self, adv_context_images, context_images, context_labels, target_images, target_labels_oh, c,
                  get_logits_fn, optimizer):
        """
        Optimize for one step.

        :param model: the model to attack
        :type model: nn.Module
        :param optimizer: the Adam optimizer to optimize ``modifier_var``
        :type optimizer: optim.Adam
        :param inputs_tanh_var: the input images in tanh-space
        :type inputs_tanh_var: Tensor
        :param pert_tanh_var: the perturbation to optimize in tanh-space,
               ``pert_tanh_var.requires_grad`` flag must be set to True
        :type pert_tanh_var: Tensor
        :param target_labels_oh: the one-hot encoded target tensor (the attack
               targets if self.targeted else image labels)
        :type target_labels_oh: Tensor
        :param c: the constant :math:`c` for each (entire) context set
        :type c: float
        :return: the batch loss, squared L2-norm of adversarial perturbations
                 (of dimension [num_context_images]), the perturbed activations (of dimension
                 [num_target_images]), the adversarial examples (of dimension [num_context_images x C x H x W])
        """
        # Logits (for the target_images), when given the adversarial context set
        pert_outputs = fix_logits(get_logits_fn(adv_context_images, context_labels, target_images))
        # Will be zero for the clean context images
        perts_norm = torch.pow(adv_context_images - context_images, 2)
        perts_norm = torch.sum(perts_norm.view(
            perts_norm.size(0), -1), 1)

        ## target_active is a tensor of dimension [B], where the i-th entry is the logit value
        ## of the correct/target class for the i-th image
        # In Carlini's code, `target_active` is called `real`.
        target_active = torch.sum(target_labels_oh * pert_outputs, 1)
        inf = 1e4  # sadly pytorch does not work with np.inf;
        # 1e4 is also used in Carlini's code

        ## Get the next most likely class i.e. for each image, find the maximum logit value
        ## that isn't the correct/target class's logit
        # In Carlini's code, `max_other_active` is called `other`.
        # max_other_active should be a Tensor of dimension [B]
        #
        # The assertion here ensures (sufficiently yet not necessarily) the
        # assumption behind the trick to get `max_other_active` holds, that
        # $\max_{i \ne t}{o_i} \ge -\text{_inf}$, where $t$ is the target and
        # $o_i$ the $i$th element along axis=1 of `pert_outputs_var`.
        #
        # noinspection PyArgumentList
        assert (pert_outputs.max(1)[0] >= -inf).all(), 'assumption failed'
        # noinspection PyArgumentList
        max_other_active = torch.max(((1.0 - target_labels_oh) * pert_outputs
                                      - target_labels_oh * inf), 1)[0]

        # Compute $f(x')$, where $x'$ is the adversarial example in image space.
        # The result `f_eval` should be of dimension [B]
        if self.targeted:
            # if targeted, optimize to make `target_active` larger than
            # `max_other_active` by `self.confidence`
            f_eval = torch.clamp(max_other_active - target_active
                                 + self.confidence, min=0.0)
        else:
            # if not targeted, optimize to make `max_other_active` larger than
            # `target_active` (the ground truth image labels) by
            # `self.confidence`
            f_eval = torch.clamp(target_active - max_other_active
                                 + self.confidence, min=0.0)
        # the total loss of current batch, should be of dimension [1]
        # The f_eval contrib is weighted by c and distributed across all dims
        combined_loss = torch.sum(perts_norm + c* torch.mean(f_eval))

        # Do optimization for one step
        optimizer.zero_grad()
        combined_loss.backward()
        optimizer.step()

        # Make some records in python/numpy on CPU
        loss = combined_loss.item()
        return loss, perts_norm, pert_outputs, adv_context_images

    def _attack_successful(self, prediction, target):
        """
        See whether the underlying attack is successful.

        :param prediction: the prediction of the model on an input
        :type prediction: int
        :param target: either the attack target or the ground-truth image label
        :type target: int
        :return: ``True`` if the attack is successful
        :rtype: bool
        """
        # Make the successfulness of an attack depend on the current iteration
        #accept_success = min(int((curr_step/self.max_iterations)*10)/10.0 + 0.1, 1.0)
        if self.targeted:
            return ((prediction == target).sum()/float(prediction.shape[0])).item() >= self.success_fraction
        else:
            return ((prediction != target).sum()/float(prediction.shape[0])).item() >= self.success_fraction

    def _compensate_confidence(self, outputs, targets):
        """
        Compensate for ``self.confidence`` and returns a new weighted sum
        vector.

        :param outputs: the weighted sum right before the last layer softmax
               normalization, of dimension [B x M]
        :type outputs: Tensor
        :param targets: either the attack targets or the real image labels,
               depending on whether or not ``self.targeted``, of dimension [B]
        :type targets: Tensor
        :return: the compensated weighted sum of dimension [B x M]
        :rtype: Tensor
        """
        outputs_comp = outputs.clone()
        rng = np.arange(targets.shape[0])
        if self.targeted:
            # for each image $i$:
            # if targeted, `outputs[i, target_onehot]` should be larger than
            # `max(outputs[i, ~target_onehot])` by `self.confidence`
            outputs_comp[rng, targets] -= self.confidence
        else:
            # for each image $i$:
            # if not targeted, `max(outputs[i, ~target_onehot]` should be larger
            # than `outputs[i, target_onehot]` (the ground truth image labels)
            # by `self.confidence`
            outputs_comp[rng, targets] += self.confidence
        return outputs_comp

    def get_attack_mode(self):
        return self.attack_mode

    def _to_tanh_space(self, x):
        """
        Convert a batch of tensors to tanh-space.

        :param x: the batch of tensors, of dimension [B x C x H x W]
        :return: the batch of tensors in tanh-space, of the same dimension
        """
        return to_tanh_space(x, self.box)

    def _from_tanh_space(self, x):
        """
        Convert a batch of tensors from tanh-space to input space.

        :param x: the batch of tensors, of dimension [B x C x H x W]
        :return: the batch of tensors in tanh-space, of the same dimension;
                 the returned tensor is on the same device as ``x``
        """
        return from_tanh_space(x, self.box)
