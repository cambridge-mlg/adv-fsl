import torch
import math
from PIL import Image
import numpy as np
import os
import pickle

class AdversarialDataset:
    def __init__(self, pickle_file_path):
        fin = open(pickle_file_path, 'rb')
        task_dict_list = pickle.load(fin)
        fin.close()
        assert len(task_dict_list) > 0
        self.tasks = task_dict_list

        self.shot = task_dict_list[0]['shot']
        self.way = task_dict_list[0]['way']
        self.query = task_dict_list[0]['query']
        self.mode = task_dict_list[0]['mode']
        self.device = task_dict_list[0]['context_images'][0].device

    def get_clean_task(self, task_index):
        context_labels = self.tasks[task_index]['context_labels'].type(torch.LongTensor).to(self.device)
        return self.tasks[task_index]['context_images'], context_labels, self.tasks[task_index]['target_images'], self.tasks[task_index]['target_labels']

    def get_adversarial_task(self, task_index):
        context_labels = self.tasks[task_index]['context_labels'].type(torch.LongTensor).to(self.device)
        return self.tasks[task_index]['adv_images'], context_labels, self.tasks[task_index]['target_images'], self.tasks[task_index]['target_labels']

    def get_eval_task(self, task_index):
        eval_labels = self.tasks[task_index]['eval_labels']
        for i in range(len(eval_labels)):
            eval_labels[i] = eval_labels[i].type(torch.LongTensor).to(self.device)
        return self.tasks[task_index]['eval_images'], eval_labels

    def get_num_tasks(self):
        return len(self.tasks)

    def get_way(self):
        return self.way

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


def distance_linf(x1, x2):
    return torch.max(torch.abs(x1 - x2), dim=1)


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


def split_target_set(target_images, target_labels, shot):
    classes = torch.unique(target_labels)
    way = len(classes)

    new_target_size = shot * way
    num_sets = int(len(target_images)/new_target_size)
    split_indices = []
    for s in range(num_sets):
        split_indices.append([])

    for c in range(way):
        c_indices_tensor = extract_class_indices(target_labels, c)
        c_indices = [val.item() for val in c_indices_tensor]

        for s in range(num_sets):
            split_indices[s].extend(c_indices[ s*shot : (s+1)*shot ])

    # So now we have num_sets-many lists of indices. Each list corresponds to the indices for a different target set.
    split_target_images = []
    split_target_labels = []
    for s in range(num_sets):
        split_target_images.append(target_images[split_indices[s]])
        split_target_labels.append(target_labels[split_indices[s]])

    return split_target_images, split_target_labels


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


def save_image(image_array, save_path):
    image_array = image_array.transpose([1, 2, 0])
    mode = 'RGB'
    if image_array.shape[2] == 1:  # single channel image
        image_array = image_array.squeeze()
        mode = 'L'
    im = Image.fromarray(np.clip((image_array + 1.0) * 127.5 + 0.5, 0, 255).astype(np.uint8), mode=mode)
    im.save(save_path)


class Logger():
    def __init__(self, checkpoint_dir, log_file_name):
        log_file_path = os.path.join(checkpoint_dir, log_file_name)
        self.file = None
        if os.path.isfile(log_file_path):
            self.file = open(log_file_path, "a", buffering=1)
        else:
            self.file = open(log_file_path, "w", buffering=1)

    def __del__(self):
        self.file.close()

    def print_and_log(self, message):
        print(message, flush=True)
        self.file.write(message + '\n')
