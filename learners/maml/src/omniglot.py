import numpy as np
import torch
import torchvision.transforms as transforms
import os

"""
   Supporting methods for data handling
"""


def extract_data(data, augment_data):
    images, char_nums = [], []
    if augment_data:
        for character in data:
            data = augment_character_set(data, character)
    for character_index, character in enumerate(data):
        for m, instance in enumerate(character):
            images.append(instance[0])
            char_nums.append(character_index)
    # each omniglot image has shape (H,W), C is implicitly 1
    # pytorch wants to process images in (C, H, W) format, so we need expand the array for the channels component
    # final images array has shape (N, C, H, W) where N is image count
    images = np.expand_dims(np.array(images), -1)
    images = np.array(images)
    char_number = np.array(char_nums)
    return images, char_number


def augment_character_set(data, character_set):
    """
    :param data: Dataset the character belongs to.
    :param character_set: np array containing instances of a character.
    :return: Original data with added character sets for all defined permutations of the current character.
    """
    rotation_90, rotation_180, rotation_270 = [], [], []
    for instance in character_set:
        image, char_num, char_language_num = instance
        rotation_90.append((np.rot90(image, k=1), char_num, char_language_num))
        rotation_180.append((np.rot90(image, k=2), char_num, char_language_num))
        rotation_270.append((np.rot90(image, k=3), char_num, char_language_num))
    return np.vstack((data, np.array([rotation_90, rotation_180, rotation_270])))


class OmniglotData(object):
    """
        Class to handle Omniglot data set. Loads from numpy data as saved in
        data folder.
    """
    def __init__(self, path, train_size=1100, validation_size=100, augment_data=True, seed=111):
        """
        Initialize object to handle Omniglot data
        :param path: directory of numpy file with preprocessed Omniglot arrays.
        :param train_size: Number of characters in training set.
        :param validation_size: Number of characters in validation set.
        :param augment_data: Augment with rotations of characters (boolean).
        :param seed: random seed for train/validation/test split.
        """
        np.random.seed(seed)
        data = np.load(os.path.join(path, 'omniglot.npy'), encoding='bytes', allow_pickle=True)
        np.random.shuffle(data)
        self.instances_per_char = 20
        self.image_height = 28
        self.image_width = 28
        self.image_channels = 1
        self.total_chars = data.shape[0]

        self.train_images, self.train_char_nums = extract_data(data[:train_size], augment_data=augment_data)
        if validation_size is not 0:
            self.validation_images, self.validation_char_nums =\
                extract_data(data[train_size:train_size + validation_size], augment_data=augment_data)
        self.test_images, self.test_char_nums =\
            extract_data(data[train_size + validation_size:], augment_data=augment_data)

        self.basic_transform = transforms.Compose([
            transforms.ToTensor()
        ])

    def get_train_task(self, way, shot, target_shot):
        return self._generate_task(images=self.train_images,
                                   character_indices= self.train_char_nums,
                                   shot=shot,
                                   way=way,
                                   eval_samples=target_shot)

    def get_validation_task(self, way, shot, target_shot):
        return self._generate_task(images=self.validation_images,
                                   character_indices=self.validation_char_nums,
                                   shot=shot,
                                   way=way,
                                   eval_samples=target_shot)

    def get_test_task(self, way, shot, target_shot):
        return self._generate_task(images=self.test_images,
                                   character_indices=self.test_char_nums,
                                   shot=shot,
                                   way=way,
                                   eval_samples=target_shot)

    def _generate_task(self, images, character_indices, shot, way, eval_samples):
        """
        Randomly generate a task from image set.
        :param images: images set to generate batch from.
        :param character_indices: indices of each character.
        :param shot: number of training images per class.
        :param way: number of classes per task.
        :param eval_samples: number of evaluation samples to use.
        :return: tuple containing train and test images and labels for a task.
        """
        num_test_instances = eval_samples
        train_images_list, test_images_list = [], []
        task_characters = np.random.choice(np.unique(character_indices), way)
        for character in task_characters:
            character_images = images[np.where(character_indices == character)[0]]
            np.random.shuffle(character_images)
            train_images_list.append(character_images[:shot])
            test_images_list.append(character_images[shot:shot + eval_samples])
        train_images, test_images = np.vstack(train_images_list), np.vstack(test_images_list)
        train_labels = np.arange(way).repeat(shot, 0)
        test_labels = np.arange(way).repeat(num_test_instances, 0)

        train_shape = train_images.shape
        train_images_tensor = torch.empty(
            size=(train_shape[0], self.image_channels, self.image_height, self.image_width),
            dtype=torch.float)
        test_shape = test_images.shape
        test_images_tensor = torch.empty(
            size=(test_shape[0], self.image_channels, self.image_height, self.image_width),
            dtype=torch.float)

        # convert images to pytorch tensors
        for i in range(train_shape[0]):
            train_images_tensor[i] = self.basic_transform(train_images[i])

        for i in range(test_shape[0]):
            test_images_tensor[i] = self.basic_transform(test_images[i])

        task_dict = {
            "context_images": train_images_tensor,
            "target_images": test_images_tensor,
            "context_labels": train_labels,
            "target_labels": test_labels
        }

        return task_dict
