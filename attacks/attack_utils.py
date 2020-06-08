import torch
import math


def convert_labels(predictions):
    return torch.argmax(predictions, dim=1, keepdim=False)


def fix_logits(logits):
    # cnaps puts an extra sampling dimension on the logits - get rid of it
    size = logits.size()
    if len(size) == 3 and size[0] == 1:
        logits = torch.squeeze(logits, dim=0)
    return logits


def generate_context_attack_indices(class_labels, class_fraction, shot_fraction):
    '''
        Given the class labels, generate the indices of the patterns we want to attack.
        We choose patterns based on the specified fraction of classes and shots to attack.
        eg. For class_fraction = 0.5, we will generate attacks for the (first) half of the classes
        For shot_fraction = 0.25, one quarter of the context set images for a particular class will be adversarial
    '''
    indices = []
    classes = torch.unique(class_labels)
    num_classes = len(classes)
    num_classes_to_attack = max(1, math.ceil(class_fraction * num_classes))
    for c in range(num_classes_to_attack):
        shot_indices = extract_class_indices(class_labels, c)
        num_shots_in_class = len(shot_indices)
        num_shots_to_attack = max(1, math.ceil(shot_fraction * num_shots_in_class))
        attack_indices = shot_indices[0:num_shots_to_attack]
        for index in attack_indices:
            indices.append(index)
    return indices


def distance_l2_squared(x1, x2):
    return torch.sum(torch.pow(x1 - x2, 2).view(x1.shape[0], -1), dim=1)


def distance_l1(x1, x2):
    return torch.sum(torch.abs(x1 - x2).view(x1.shape[0], -1), dim=1)


def extract_class_indices(labels, which_class):
    """
    Helper method to extract the indices of elements which have the specified label.
    :param labels: (torch.tensor) Labels of the context set.
    :param which_class: Label for which indices are extracted.
    :return: (torch.tensor) Indices in the form of a mask that indicate the locations of the specified label.
    """
    class_mask = torch.eq(labels, which_class)  # binary mask of labels equal to which_class
    class_mask_indices = torch.nonzero(class_mask)  # indices of labels equal to which class
    return torch.reshape(class_mask_indices, (-1,))  # reshape to be a 1D vector


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