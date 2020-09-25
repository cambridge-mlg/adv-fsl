import torch
import math
from PIL import Image
import numpy as np
import os
import pickle
import bz2
import _pickle as cPickle


def save_pickle(file_path, data, compress=False):
    if compress:
        file_path += ".pbz2"
        with bz2.BZ2File(filename=file_path, mode='w') as f:
            cPickle.dump(data, f)
    else:
        file_path += ".pickle"
        f = open(file_path, 'wb')
        pickle.dump(data, f)
        f.close()


def load_pickle(file_path):
    extension = os.path.splitext(file_path)[1]
    if extension == '.pbz2':
        data = bz2.BZ2File(file_path, 'rb')
        data = cPickle.load(data)
    else:
        f = open(file_path, 'rb')
        data = pickle.load(f)
        f.close()
    return data


class AdversarialDataset:
    def __init__(self, pickle_file_path):
        task_dict_list = load_pickle(pickle_file_path)
        assert len(task_dict_list) > 0
        self.tasks = task_dict_list

        self.shot = task_dict_list[0]['shot']
        self.way = task_dict_list[0]['way']
        self.query = task_dict_list[0]['query']
        mode = task_dict_list[0]['mode']
        assert mode == 'context' or mode == 'target' or mode == 'swap'
        self.mode = mode

    def get_clean_task(self, task_index, device):
        context_labels = self.tasks[task_index]['context_labels'].type(torch.LongTensor).to(device)
        return self.tasks[task_index]['context_images'].to(device), context_labels, self.tasks[task_index]['target_images'].to(device), self.tasks[task_index]['target_labels'].to(device)

    def get_adversarial_task(self, task_index, device, swap_mode=None):
        context_labels = self.tasks[task_index]['context_labels'].type(torch.LongTensor).to(device)
        if self.mode == 'context':
            assert swap_mode is None
            return self.tasks[task_index]['adv_images'].to(device), context_labels, self.tasks[task_index]['target_images'].to(device), self.tasks[task_index]['target_labels'].to(device)
        elif self.mode == 'target':
            assert swap_mode is None
            return self.tasks[task_index]['context_images'].to(device), context_labels, self.tasks[task_index]['adv_images'].to(device), self.tasks[task_index]['target_labels'].to(device)
        elif self.mode == 'swap':
            assert swap_mode is not None
            if swap_mode == 'context':
                return self.tasks[task_index]['adv_context_images'].to(device), context_labels, self.tasks[task_index][
                    'target_images'].to(device), self.tasks[task_index]['target_labels'].to(device)
            else:
                return self.tasks[task_index]['context_images'].to(device), context_labels, self.tasks[task_index][
                    'adv_target_images'].to(device), self.tasks[task_index]['target_labels'].to(device)

    def get_eval_task(self, task_index, device):
        eval_labels = self.tasks[task_index]['eval_labels']
        eval_images = self.tasks[task_index]['eval_images']
        eval_images_gpu = []
        eval_labels_gpu = []
        for i in range(len(eval_labels)):
            eval_labels_gpu.append(eval_labels[i].type(torch.LongTensor).to(device))
            eval_images_gpu.append(eval_images[i].to(device))
        return eval_images_gpu, eval_labels_gpu

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


def generate_attack_indices(class_labels, class_fraction, shot_fraction):
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
            indices.append(index.item())
    return indices

def infer_num_shots(class_labels):
    classes = torch.unique(class_labels)
    num_classes = len(classes)
    num_shots = -1
    for c in range(num_classes):
        num_shots_curr = len(extract_class_indices(class_labels, c))
        if num_shots_curr != num_shots and num_shots != -1:
            print("Expected all classes to have equal number of shots")
            return -1
        elif num_shots == -1:
            num_shots = num_shots_curr
    return num_shots


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


def get_random_targeted_labels(true_labels, device):
    classes = torch.unique(true_labels)
    way = len(classes)
    targeted_labels = torch.randint(1, way, (len(true_labels),)).to(device)
    targeted_labels = torch.fmod(targeted_labels + true_labels, way)
    return targeted_labels


def get_shifted_targeted_labels(true_labels, device):
    classes = torch.unique(true_labels)
    way = len(classes)
    targeted_labels = torch.fmod(true_labels + torch.ones_like(true_labels), way)
    return targeted_labels

def infer_way_and_shots(labels):
    classes = torch.unique(labels)
    way = len(classes)

    shots = []
    for c in range(way):
        c_indices = extract_class_indices(labels, c)
        shots.append(c_indices.shape[0])
    return way, shots

def split_into_tasks(target_images, target_labels, shot, target_images_np=None):
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

    if target_images_np is not None:
        split_target_images_np = []
        for s in range(num_sets):
            split_target_images_np.append(target_images_np[split_indices[s]])
        return split_target_images, split_target_labels, split_target_images_np
    else:
        return split_target_images, split_target_labels, None


def split_target_set(all_target_images, all_target_labels, target_set_size_multiplier, shot, all_target_images_np=None, return_first_target_set=False):
    split_target_images, split_target_labels, split_target_images_np = split_into_tasks(all_target_images, all_target_labels, shot, target_images_np=all_target_images_np)

    # The first "target_set_size_multiplier"-many will be used when generating the attack
    # The rest will be used for independent eval
    # Note that we have to split them first, because this ensures each block has equal class representation
    eval_start_index = target_set_size_multiplier
    target_images = torch.stack(split_target_images[0:eval_start_index]).view(-1, all_target_images.shape[1],
                                                                              all_target_images.shape[2],
                                                                              all_target_images.shape[3])
    target_labels = torch.stack(split_target_labels[0:eval_start_index]).view(-1)

    if all_target_images_np is None:
        # No additional info required except the target set and the eval sets
        if not return_first_target_set:
            return target_images, target_labels, split_target_images[eval_start_index:], split_target_labels[eval_start_index:]
        # Caller requires a "standard" size target set as well
        else:
            return target_images, target_labels, split_target_images[eval_start_index:], split_target_labels[eval_start_index:], \
                   split_target_images[0], split_target_labels[0]
    else:
        #Caller requires that we split the numpy version of the target set as well
        target_images_np = np.concatenate(split_target_images_np[0:eval_start_index], axis=0)
        if not return_first_target_set:
            return target_images, target_labels, split_target_images[eval_start_index:], split_target_labels[eval_start_index:], target_images_np,
        # Target requires split numpy version as well as "standard" size target set
        else:
            return target_images, target_labels, split_target_images[eval_start_index:], split_target_labels[eval_start_index:], target_images_np, split_target_images[0], split_target_labels[0]



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


def save_image(image_array, save_path, scaling='neg_one_to_one'):
    image_array = image_array.transpose([1, 2, 0])
    mode = 'RGB'
    if image_array.shape[2] == 1:  # single channel image
        image_array = image_array.squeeze()
        mode = 'L'
    if scaling == 'neg_one_to_one':
        im = Image.fromarray(np.clip((image_array + 1.0) * 127.5 + 0.5, 0, 255).astype(np.uint8), mode=mode)
    else:
        im = Image.fromarray(np.clip(image_array * 255.0, 0, 255).astype(np.uint8), mode=mode)
    im.save(save_path)

def make_adversarial_task_dict(context_images, context_labels, target_images, target_labels, adv_images, adv_indices, attack_mode, way, shot, query, split_target_images, split_target_labels):
    adv_task_dict = {
        'context_images': context_images.cpu(),
        'context_labels': context_labels.cpu(),
        'target_images': target_images.cpu(),
        'target_labels': target_labels.cpu(),
        'adv_images': adv_images.cpu(),
        'adv_indices': adv_indices,
        'mode': attack_mode,
        'way': way,
        'shot': shot,
        'query': query,
    }
    if split_target_images is not None and split_target_labels is not None:
        adv_task_dict['eval_images'] = []
        adv_task_dict['eval_labels'] = []
        for k in range(len(split_target_images)):
            adv_task_dict['eval_images'].append(split_target_images[k].cpu())
            adv_task_dict['eval_labels'].append(split_target_labels[k].cpu())
    return adv_task_dict

def make_swap_attack_task_dict(context_images, context_labels, target_images, target_labels, adv_context_images, adv_context_indices, adv_target_images, adv_target_indices,
                                way, shot, query, split_target_images, split_target_labels):
    adv_task_dict = make_adversarial_task_dict(context_images, context_labels, target_images, target_labels, adv_context_images, adv_context_indices, 'context', way, shot, query, split_target_images, split_target_labels)
    adv_task_dict['adv_target_images'] = adv_target_images.cpu()
    adv_task_dict['adv_target_indices'] = adv_target_indices
    adv_task_dict['adv_context_images'] = adv_task_dict['adv_images'] #This should just reference the same tensor, hopefully not make a whole copy
    adv_task_dict['adv_context_indices'] = adv_task_dict['adv_indices']
    adv_task_dict['mode'] = 'swap'
    return adv_task_dict


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
