import torch
from  torch.distributions.uniform import Uniform
from attacks.attack_utils import generate_context_attack_indices, Logger


class RandomAttack:
    def __init__(self,
                 checkpoint_dir,
                 attack_mode='context',
                 class_fraction=1.0,
                 shot_fraction=1.0,
                 epsilon=0.1,
                 normalize_perturbation=True):
        self.attack_mode = attack_mode
        self.class_fraction = class_fraction
        self.shot_fraction = shot_fraction
        self.epsilon = epsilon
        self.logger = Logger(checkpoint_dir, "random_logs.txt")
        self.verbose = False
        self.normalize_perturbation = normalize_perturbation

    # Epsilon is specified for inputs normalized to [0,1].
    # Use a sample of the images to recalculate the required perturbation size (for actual image normalization)
    def normalize_epsilon(self, clip_min, clip_max):
        epsilon_new = self.epsilon
        if clip_min != 0.0 or clip_max != 1.0:
            # Epsilon is specified relative to min = 0, max = 1.0
            # If this is not the case, scale epsilon
            epsilon_new = self.epsilon * (clip_max - clip_min)
        return epsilon_new

    def generate(self, context_images, context_labels, target_images, true_target_labels, model, get_logits_fn, device,
                 targeted_labels=None):
        self.logger.print_and_log("Performing Random attack on {} set. Settings = (epsilon={})"
                                  .format(self.attack_mode, self.epsilon))
        self.logger.print_and_log("class_fraction = {}, shot_fraction = {}"
                                  .format(self.class_fraction, self.shot_fraction))
        self.logger.print_and_log("context set size = {}, target set size = {}"
                                  .format(context_images.shape[0], target_images.shape[0]))

        if self.attack_mode == 'target':
            return self._generate_target(context_images, context_labels, target_images, None, model, get_logits_fn,
                                         device, targeted_labels=targeted_labels)
        else:  # context
            return self._generate_context(context_images, context_labels, target_images, None, model, get_logits_fn,
                                          device, targeted_labels=targeted_labels)

    def _generate_target(self, context_images, context_labels, target_images, target_labels, model, get_logits_fn, device, targeted_labels=None):
        clip_min = target_images.min().item()
        clip_max = target_images.max().item()
        # Desired epsilon are given for images normalized to [0,1]. Adjust for when this is not the case
        if self.normalize_perturbation:
            epsilon = self.normalize_epsilon(clip_min, clip_max)
        else:
            epsilon = self.epsilon
        self.logger.print_and_log("Normalized perturbation sizes: eps={}".format(epsilon))

        adv_target_images = target_images.clone()

        # initialize uniform noise
        noise = Uniform(torch.tensor([-epsilon]), torch.tensor([epsilon]))
        noise_shape = adv_target_images.size()

        adv_target_images = torch.clamp(adv_target_images + noise.rsample(noise_shape).squeeze().to(device), clip_min, clip_max)

        return adv_target_images, list(range(adv_target_images.shape[0]))

    def _generate_context(self, context_images, context_labels, target_images, target_labels, model, get_logits_fn, device, targeted_labels=None):
        clip_min = target_images.min().item()
        clip_max = target_images.max().item()
        if self.normalize_perturbation:
            epsilon = self.normalize_epsilon(clip_min, clip_max)
        else:
            epsilon = self.epsilon

        adv_context_indices = generate_context_attack_indices(context_labels, self.class_fraction, self.shot_fraction)
        adv_context_images = context_images.clone()

        # initialize uniform noise
        noise = Uniform(torch.tensor([-epsilon / 2.0]), torch.tensor([epsilon / 2.0]))
        shape = adv_context_images.size()
        noise_shape = [shape[1], shape[2], shape[3]]

        for index in adv_context_indices:
            adv_context_images[index] = torch.clamp(adv_context_images[index] + noise.rsample(noise_shape).squeeze().to(device), clip_min, clip_max)

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
