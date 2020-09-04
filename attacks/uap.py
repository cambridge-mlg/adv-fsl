import torch
import numpy as np
from attacks.attack_utils import convert_labels, generate_context_attack_indices, fix_logits, Logger


class UapAttack:
    def __init__(self,
                 checkpoint_dir,
                 norm='inf',
                 attack_mode='context',
                 class_fraction=1.0,
                 shot_fraction=1.0,
                 perturbation_image_path=None):
        self.attack_mode = attack_mode
        self.class_fraction = class_fraction
        self.shot_fraction = shot_fraction
        self.logger = Logger(checkpoint_dir, "uap_logs.txt")
        self.verbose = False

        # load perturbation image
        uap_image_np = np.load(perturbation_image_path)
        uap_image_np = uap_image_np.transpose([2, 0, 1])
        self.uap_image = torch.from_numpy(uap_image_np).type(torch.FloatTensor)

    def generate(self, context_images, context_labels, target_images, true_target_labels, model, get_logits_fn, device, targeted_labels=None):
        self.logger.print_and_log("Performing UAP attack on {} set.".format(self.attack_mode))
        self.logger.print_and_log("class_fraction = {}, shot_fraction = {}".format(self.class_fraction, self.shot_fraction))
        self.logger.print_and_log("context set size = {}, target set size = {}".format(context_images.shape[0], target_images.shape[0]))
        self.uap_image = self.uap_image.to(device)

        if self.attack_mode == 'target':
            return self._generate_target(context_images, context_labels, target_images, None, model, get_logits_fn,
                                         device, targeted_labels=targeted_labels)
        else:  # context
            return self._generate_context(context_images, context_labels, target_images, None, model, get_logits_fn,
                                          device, targeted_labels=targeted_labels)

    def _generate_target(self, context_images, context_labels, target_images, target_labels, model, get_logits_fn, device, targeted_labels=None):
        adv_target_images = target_images.clone()

        # shift and scale the images to the 0 - 255 range
        adv_target_images = (adv_target_images + 1.0) * 127.5
        # add the uap image
        for i in range(adv_target_images.size(0)):
            adv_target_images[i] = torch.clamp(adv_target_images[i] + self.uap_image, 0.0, 255.0)
        # scale back to -1 to 1
        adv_target_images = (adv_target_images / 127.5) - 1.0

        return adv_target_images, list(range(adv_target_images.shape[0]))

    def _generate_context(self, context_images, context_labels, target_images, target_labels, model, get_logits_fn, device, targeted_labels=None):
        adv_context_indices = generate_context_attack_indices(context_labels, self.class_fraction, self.shot_fraction)
        adv_context_images = context_images.clone()

        for index in adv_context_indices:
            # shift and scale the images to the 0 - 255 range
            adv_context_images[index] = (adv_context_images[index] + 1.0) * 127.5
            # add the uap image
            adv_context_images[index] = torch.clamp(adv_context_images[index] + self.uap_image, 0.0, 255.0)
            # scale back to -1 to 1
            adv_context_images[index] = (adv_context_images[index] / 127.5) - 1.0

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
