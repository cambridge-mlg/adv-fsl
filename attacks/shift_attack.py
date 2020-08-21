import torch
import math
import numpy as np
from attacks.attack_utils import extract_class_indices, Logger


class ShiftAttack:
    def __init__(self,
                 checkpoint_dir,
                 attack_mode='context',
                 class_fraction=1.0,
                 shot_fraction=1.0,
                 ):
        self.attack_mode = attack_mode
        self.class_fraction = class_fraction
        self.shot_fraction = shot_fraction
        self.logger = Logger(checkpoint_dir, "shift_logs.txt")

        self.verbose = False

    def generate(self, context_images, context_labels, target_images, true_target_labels, model, get_logits_fn, device, targeted_labels=None):
        self.logger.print_and_log("Performing PGD attack on {} set.".format(self.attack_mode))
        self.logger.print_and_log("class_fraction = {}, shot_fraction = {}".format(self.class_fraction, self.shot_fraction))
        self.logger.print_and_log("context set size = {}, target set size = {}".format(context_images.shape[0], target_images.shape[0]))

        if self.attack_mode == 'target':
            labels = true_target_labels
            imgs = target_images
        else:
            labels = context_labels
            imgs = context_images

        num_classes = torch.unique(labels)
        shot = extract_class_indices(labels, 0)
        # Make sure that all classes have equal shot, which we require for this attack to work.
        # (We can do it otherwise, but that would require some adjusting in the calling functions)
        for c in range(1, num_classes):
            c_shot_indices = extract_class_indices(labels, c)
            assert len(c_shot_indices) == shot
        num_adv_shots = max(1, math.ceil(self.shot_fraction * shot))

        num_adv_classes = max(1, math.ceil(self.class_fraction * num_classes))
        # We need to attack at least two classes to be able to do a shift attack
        assert num_adv_classes > 0

        adv_imgs_list = []
        adv_indices = []
        label_shift = []
        for c in range(num_classes):
            # Adversarial classes, shift
            if c < num_adv_classes:
                shot_indices = extract_class_indices(labels, c)
                shifted_class = (c + 1) % num_adv_classes
                shifted_shot_indices = extract_class_indices(labels, shifted_class)
                attack_indices = shifted_shot_indices[0:num_adv_shots]
                clean_indices = shot_indices[num_adv_shots:]
                adv_index = len(adv_imgs_list)

                for index in attack_indices:
                    adv_imgs_list.append(imgs[index].clone())
                    label_shift.append(labels[index])
                    adv_indices.append(adv_index)
                    adv_index += 1

                for index in clean_indices:
                    adv_imgs_list.append(imgs[index].clone())
                    label_shift.append(labels[index])

            # Not adversarial, don't shift, just append
            else:
                shot_indices = extract_class_indices(labels, c)
                for index in shot_indices:
                    adv_imgs_list.append(imgs[index].clone())
                    label_shift.append(labels[index])

        self.logger.print_and_log("True labels {}".format(label_shift))

        adv_imgs = torch.stack(adv_imgs_list, dim=0)
        return adv_imgs, adv_indices

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
