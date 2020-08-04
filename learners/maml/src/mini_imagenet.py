import os
import numpy as np
import pickle
import torch
import torchvision.transforms as transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def shuffle_batch(images, labels):
    """
       Return a shuffled batch of data
    """
    permutation = np.random.permutation(images.shape[0])
    return images[permutation], labels[permutation]


def prepare_task(task_dict, shuffle_examples=True):
    context_images, context_labels = task_dict['context_images'], \
                                     task_dict['context_labels']
    target_images = task_dict['target_images'] 
    target_labels = task_dict['target_labels']

    if shuffle_examples:
        context_images, context_labels = shuffle(context_images,
                                                 context_labels)
        target_images, target_labels = shuffle(target_images,
                                               target_labels)

    context_images = context_images.to(device)
    target_images = target_images.to(device)
    context_labels = torch.from_numpy(context_labels).type(
        torch.LongTensor).to(device)
    target_labels = torch.from_numpy(target_labels).type(
        torch.LongTensor).to(device)

    return context_images, target_images, context_labels, target_labels


def shuffle(images, labels):
    """
    Return shuffled data.
    """
    permutation = np.random.permutation(images.shape[0])
    return images[permutation], labels[permutation]


class MiniImageNetData(object):

    def __init__(self, path, seed):
        """
        Constructs a miniImageNet dataset for use in episodic training.
        :param path: Path to miniImageNet data files.
        :param seed: Random seed to reproduce batches.
        """
        np.random.seed(seed)

        path_train = os.path.join(path, 'mini_imagenet_train.pkl')
        path_validation = os.path.join(path, 'mini_imagenet_val.pkl')
        path_test = os.path.join(path, 'mini_imagenet_test.pkl')

        self.train_set = pickle.load(open(path_train, 'rb'))
        self.validation_set = pickle.load(open(path_validation, 'rb'))
        self.test_set = pickle.load(open(path_test, 'rb'))

        self.image_width = 84
        self.image_height = 84
        self.image_channels = 3

        # normalize to -1 to 1
        normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                         std=[0.5, 0.5, 0.5])

        self.basic_transform = transforms.Compose([
            transforms.ToTensor(),
            normalize
        ])

    def get_train_task(self, way, shot, target_shot):
        return self._generate_task(images=self.train_set,
                                   shot=shot,
                                   way=way,
                                   eval_samples=target_shot)

    def get_validation_task(self, way, shot, target_shot):
        return self._generate_task(images=self.validation_set,
                                   shot=shot,
                                   way=way,
                                   eval_samples=target_shot)

    def get_test_task(self, way, shot, target_shot):
        return self._generate_task(images=self.test_set,
                                   shot=shot,
                                   way=way,
                                   eval_samples=target_shot)

    def _generate_task(self, images, shot, way, eval_samples):
        """
        Sample a k-shot batch from images.
        :param images: Data to sample from [n_classes, n_samples, h, w, c] (either of xTrain, xVal, xTest)
        :param tasks_per_batch: number of tasks to include in batch.
        :param shot: number of training examples per class.
        :param way: number of classes per task.
        :param eval_samples: number of samples to use in evaluation
        :return: A list [train_images, test_images, train_labels, test_labels]

        shapes:
            * Images: [tasksPerBatch, classesPerTask*samplesPerClassTrain/Test, c, h, w]
            * Labels: [tasksPerBatch, classesPerTask*samplesPerClassTrain/Test, classesPerTask]
                      (one-hot encoded in last dim)
        """
        samples_per_class = shot + eval_samples
        classes_idx = np.arange(images.shape[0])
        samples_idx = np.arange(images.shape[1])

        num_test_instances = eval_samples
        train_images_list, test_images_list = [], []

        choose_classes = np.random.choice(classes_idx, size=way, replace=False)
        for image_class in choose_classes:
            choose_samples = np.random.choice(samples_idx, size=samples_per_class, replace=False)
            temp_images = images[image_class, choose_samples, ...]
            np.random.shuffle(temp_images)
            train_images_list.append(temp_images[:shot])
            test_images_list.append(temp_images[shot:])

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

        # convert images to pytorch tensors, normalize them, and set them to the device
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

