"""
Script to reproduce the few-shot classification results on Meta-Dataset in:
"Fast and Flexible Multi-Task Classification Using Conditional Neural Adaptive Processes"
https://arxiv.org/pdf/1906.07697.pdf

The following command lines should reproduce the published results within error-bars:

Note before running any of the commands, you need to run the following two commands:

ulimit -n 50000

export META_DATASET_ROOT=<root directory of the cloned or downloaded Meta-Dataset repository>

CNAPs using auto-regressive FiLM adaptation, meta-training on all datasets
--------------------------------------------------------------------------
python run_cnaps.py --data_path <path to directory containing Meta-Dataset records>

CNAPs using FiLM adaptation only, meta-training on all datasets
---------------------------------------------------------------
python run_cnaps.py --feature_adaptation film --data_path <path to directory containing Meta-Dataset records>

CNAPs using no feature adaptation, meta-training on all datasets
----------------------------------------------------------------
python run_cnaps.py --feature_adaptation no_adaptation --data_path <path to directory containing Meta-Dataset records>

CNAPs using FiLM adaptation and TaskNorm, meta-training on all datasets
-----------------------------------------------------------------------
python run_cnaps.py --feature_adaptation film -i 60000 -lr 0.001 --batch_normalization task_norm-i
                    --data_path <path to directory containing Meta-Dataset records>

- Note that when using Meta-Dataset and auto-regressive FiLM adaptation or FiLM adaptation with TaskNorm
batch normalization, 2 GPUs with at least 16GB of memory are required.
- The other modes require only a single GPU with at least 16 GB of memory.
- If you want to run any of the modes on a single GPU, you can train on a single dataset with fixed shot and way, an
example command line is (though this will not reproduce the meta-dataset results):
python run_cnaps.py --feature_adaptation film -i 20000 -lr 0.001 --batch_normalization task_norm-i
                    -- dataset omniglot --way 5 --shot 5 --data_path <path to directory containing Meta-Dataset records>

"""
# import os
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"]="1"  # specify which GPU(s) to be used

import torch
import torch.nn as nn
import numpy as np
import argparse
import os
import pickle
from normalization_layers import TaskNormI
from utils import print_and_log, get_log_files, ValidationAccuracies, loss, aggregate_accuracy, verify_checkpoint_dir
from model import Cnaps
from art_wrapper import Art_Wrapper
from meta_dataset_reader import MetaDatasetReader, SingleDatasetReader
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Quiet TensorFlow warnings
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)  # Quiet TensorFlow warnings
# from art.attacks import ProjectedGradientDescent, FastGradientMethod
# from art.classifiers import PyTorchClassifier
from PIL import Image
from attacks import ProjectedGradientDescent

NUM_VALIDATION_TASKS = 200
NUM_TEST_TASKS = 600
PRINT_FREQUENCY = 1000


def save_image(image_array, save_path):
    image_array = image_array.squeeze()
    image_array = image_array.transpose([1, 2, 0])
    im = Image.fromarray(np.clip((image_array + 1.0) * 127.5 + 0.5, 0, 255).astype(np.uint8), mode='RGB')
    im.save(save_path)

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


def main():
    learner = Learner()
    learner.run()


class Learner:
    def __init__(self):
        self.args = self.parse_command_line()

        self.checkpoint_dir, self.logfile, self.checkpoint_path_validation, self.checkpoint_path_final \
            = get_log_files(self.args.checkpoint_dir, self.args.resume_from_checkpoint, self.args.mode == "test" or 
                            self.args.mode == "attack")

        print_and_log(self.logfile, "Options: %s\n" % self.args)
        print_and_log(self.logfile, "Checkpoint Directory: %s\n" % self.checkpoint_dir)

        gpu_device = 'cuda:0'
        self.device = torch.device(gpu_device if torch.cuda.is_available() else 'cpu')
        self.model = self.init_model()
        self.train_set, self.validation_set, self.test_set = self.init_data()
        if self.args.dataset == "meta-dataset":
            self.dataset = MetaDatasetReader(self.args.data_path, self.args.mode, self.train_set, self.validation_set,
                                             self.test_set, self.args.max_way_train, self.args.max_way_test,
                                             self.args.max_support_train, self.args.max_support_test)
        else:
            self.dataset = SingleDatasetReader(self.args.data_path, self.args.mode, self.args.dataset, self.args.way,
                                               self.args.shot, self.args.query_train, self.args.query_test)
        self.loss = loss
        self.accuracy_fn = aggregate_accuracy
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        self.validation_accuracies = ValidationAccuracies(self.validation_set)
        self.start_iteration = 0
        if self.args.resume_from_checkpoint:
            self.load_checkpoint()
        self.optimizer.zero_grad()

    def init_model(self):
        use_two_gpus = self.use_two_gpus()
        model = Cnaps(device=self.device, use_two_gpus=use_two_gpus, args=self.args).to(self.device)
        self.register_extra_parameters(model)

        # set encoder is always in train mode (it only sees context data).
        model.train()
        # Feature extractor is in eval mode by default, but gets switched in model depending on args.batch_normalization
        model.feature_extractor.eval()
        if use_two_gpus:
            model.distribute_model()
        return model

    def init_data(self):
        if self.args.dataset == "meta-dataset":
            train_set = ['ilsvrc_2012', 'omniglot', 'aircraft', 'cu_birds', 'dtd', 'quickdraw', 'fungi', 'vgg_flower']
            validation_set = ['ilsvrc_2012', 'omniglot', 'aircraft', 'cu_birds', 'dtd', 'quickdraw', 'fungi', 'vgg_flower',
                              'mscoco']
            test_set = self.args.test_datasets
        else:
            train_set = [self.args.dataset]
            validation_set = [self.args.dataset]
            test_set = [self.args.dataset]

        return train_set, validation_set, test_set

    """
    Command line parser
    """
    def parse_command_line(self):
        parser = argparse.ArgumentParser()

        parser.add_argument("--dataset", choices=["meta-dataset", "ilsvrc_2012", "omniglot", "aircraft", "cu_birds",
                                                  "dtd", "quickdraw", "fungi", "vgg_flower", "traffic_sign", "mscoco",
                                                  "mnist", "cifar10", "cifar100"], default="meta-dataset",
                            help="Dataset to use.")
        parser.add_argument('--test_datasets', nargs='+', help='Datasets to use for testing',
                            default=["ilsvrc_2012", "omniglot", "aircraft", "cu_birds", "dtd", "quickdraw", "fungi",
                                     "vgg_flower", "traffic_sign", "mscoco", "mnist", "cifar10", "cifar100"])
        parser.add_argument("--data_path", default="../datasets", help="Path to dataset records.")
        parser.add_argument("--pretrained_resnet_path", default="../models/pretrained_resnet.pt.tar",
                            help="Path to pretrained feature extractor model.")
        parser.add_argument("--mode", choices=["train", "test", "train_test", "attack"], default="train_test",
                            help="Whether to run training only, testing only, or both training and testing.")
        parser.add_argument("--learning_rate", "-lr", type=float, default=5e-4, help="Learning rate.")
        parser.add_argument("--tasks_per_batch", type=int, default=16,
                            help="Number of tasks between parameter optimizations.")
        parser.add_argument("--checkpoint_dir", "-c", default='../checkpoints', help="Directory to save checkpoint to.")
        parser.add_argument("--test_model_path", "-m", default=None, help="Path to model to load and test.")
        parser.add_argument("--feature_adaptation", choices=["no_adaptation", "film", "film+ar"], default="film",
                            help="Method to adapt feature extractor parameters.")
        parser.add_argument("--batch_normalization", choices=["basic",  "task_norm-i"],
                            default="basic", help="Normalization layer to use.")
        parser.add_argument("--training_iterations", "-i", type=int, default=110000,
                            help="Number of meta-training iterations.")
        parser.add_argument("--attack_tasks", "-a", type=int, default=10,
                            help="Number of tasks when performing attack.")
        parser.add_argument("--val_freq", type=int, default=10000, help="Number of iterations between validations.")
        parser.add_argument("--max_way_train", type=int, default=40,
                            help="Maximum way of meta-dataset meta-train task.")
        parser.add_argument("--max_way_test", type=int, default=50, help="Maximum way of meta-dataset meta-test task.")
        parser.add_argument("--max_support_train", type=int, default=400,
                            help="Maximum support set size of meta-dataset meta-train task.")
        parser.add_argument("--max_support_test", type=int, default=500,
                            help="Maximum support set size of meta-dataset meta-test task.")
        parser.add_argument("--resume_from_checkpoint", "-r", dest="resume_from_checkpoint", default=False,
                            action="store_true", help="Restart from latest checkpoint.")
        parser.add_argument("--way", type=int, default=5, help="Way of single dataset task.")
        parser.add_argument("--shot", type=int, default=1, help="Shots per class for context of single dataset task.")
        parser.add_argument("--query_train", type=int, default=10,
                            help="Shots per class for target  of single dataset task.")
        parser.add_argument("--query_test", type=int, default=10,
                            help="Shots per class for target  of single dataset task.")

        args = parser.parse_args()

        return args

    def run(self):
        config = tf.compat.v1.ConfigProto()
        config.gpu_options.allow_growth = True
        with tf.compat.v1.Session(config=config) as session:
            if self.args.mode == 'train' or self.args.mode == 'train_test':
                train_accuracies = []
                losses = []
                total_iterations = self.args.training_iterations
                for iteration in range(self.start_iteration, total_iterations):
                    torch.set_grad_enabled(True)
                    task_dict = self.dataset.get_train_task(session)
                    task_loss, task_accuracy = self.train_task(task_dict)
                    train_accuracies.append(task_accuracy)
                    losses.append(task_loss)

                    # optimize
                    if ((iteration + 1) % self.args.tasks_per_batch == 0) or (iteration == (total_iterations - 1)):
                        self.optimizer.step()
                        self.optimizer.zero_grad()

                    if (iteration + 1) % PRINT_FREQUENCY == 0:
                        # print training stats
                        print_and_log(self.logfile,'Task [{}/{}], Train Loss: {:.7f}, Train Accuracy: {:.7f}'
                                      .format(iteration + 1, total_iterations, torch.Tensor(losses).mean().item(),
                                              torch.Tensor(train_accuracies).mean().item()))
                        train_accuracies = []
                        losses = []

                    if ((iteration + 1) % self.args.val_freq == 0) and (iteration + 1) != total_iterations:
                        # validate
                        accuracy_dict = self.validate(session)
                        self.validation_accuracies.print(self.logfile, accuracy_dict)
                        # save the model if validation is the best so far
                        if self.validation_accuracies.is_better(accuracy_dict):
                            self.validation_accuracies.replace(accuracy_dict)
                            torch.save(self.model.state_dict(), self.checkpoint_path_validation)
                            print_and_log(self.logfile, 'Best validation model was updated.')
                            print_and_log(self.logfile, '')
                        self.save_checkpoint(iteration + 1)

                # save the final model
                torch.save(self.model.state_dict(), self.checkpoint_path_final)

            if self.args.mode == 'train_test':
                self.test(self.checkpoint_path_final, session)
                self.test(self.checkpoint_path_validation, session)

            if self.args.mode == 'test':
                self.test(self.args.test_model_path, session)

            if self.args.mode == 'attack':
                self.attack_homebrew(self.args.test_model_path, session)
                # self.attack(self.args.test_model_path, session)

            self.logfile.close()

    def train_task(self, task_dict):
        context_images, target_images, context_labels, target_labels = self.prepare_task(task_dict)

        target_logits = self.model(context_images, context_labels, target_images)
        task_loss = self.loss(target_logits, target_labels, self.device) / self.args.tasks_per_batch
        if self.args.feature_adaptation == 'film' or self.args.feature_adaptation == 'film+ar':
            if self.use_two_gpus():
                regularization_term = (self.model.feature_adaptation_network.regularization_term()).cuda(0)
            else:
                regularization_term = (self.model.feature_adaptation_network.regularization_term())
            regularizer_scaling = 0.001
            task_loss += regularizer_scaling * regularization_term
        task_accuracy = self.accuracy_fn(target_logits, target_labels)

        task_loss.backward(retain_graph=False)

        return task_loss, task_accuracy

    def validate(self, session):
        with torch.no_grad():
            accuracy_dict ={}
            for item in self.validation_set:
                accuracies = []
                for _ in range(NUM_VALIDATION_TASKS):
                    task_dict = self.dataset.get_validation_task(item, session)
                    context_images, target_images, context_labels, target_labels = self.prepare_task(task_dict)
                    target_logits = self.model(context_images, context_labels, target_images)
                    accuracy = self.accuracy_fn(target_logits, target_labels)
                    accuracies.append(accuracy.item())
                    del target_logits

                accuracy = np.array(accuracies).mean() * 100.0
                confidence = (196.0 * np.array(accuracies).std()) / np.sqrt(len(accuracies))

                accuracy_dict[item] = {"accuracy": accuracy, "confidence": confidence}

        return accuracy_dict

    def test(self, path, session):
        print_and_log(self.logfile, "")  # add a blank line
        print_and_log(self.logfile, 'Testing model {0:}: '.format(path))
        self.model = self.init_model()
        self.model.load_state_dict(torch.load(path))

        with torch.no_grad():
            for item in self.test_set:
                accuracies = []
                for _ in range(NUM_TEST_TASKS):
                    task_dict = self.dataset.get_test_task(item, session)
                    context_images, target_images, context_labels, target_labels = self.prepare_task(task_dict)
                    target_logits = self.model(context_images, context_labels, target_images)
                    accuracy = self.accuracy_fn(target_logits, target_labels)
                    accuracies.append(accuracy.item())
                    del target_logits

                accuracy = np.array(accuracies).mean() * 100.0
                accuracy_confidence = (196.0 * np.array(accuracies).std()) / np.sqrt(len(accuracies))

                print_and_log(self.logfile, '{0:}: {1:3.1f}+/-{2:2.1f}'.format(item, accuracy, accuracy_confidence))

    def attack_homebrew(self, path, session):
        print_and_log(self.logfile, "")  # add a blank line
        print_and_log(self.logfile, 'Attacking model {0:}: '.format(path))
        self.model = self.init_model()
        self.model.load_state_dict(torch.load(path))

        fgm_attack = ProjectedGradientDescent()

        for item in self.test_set:

            for t in range(self.args.attack_tasks):

                task_dict = self.dataset.get_test_task(item, session)
                context_images, target_images, context_labels, target_labels, context_images_np = self.prepare_task(
                    task_dict, shuffle=False)

                adv_context_images, adv_context_indices = fgm_attack.generate(
                    context_images,
                    context_labels,
                    target_images,
                    self.model)

                for index in adv_context_indices:
                    save_image(adv_context_images[index].cpu().detach().numpy(),
                               os.path.join(self.checkpoint_dir, 'adv_task_{}_index_{}.png'.format(t, index)))
                    save_image(context_images_np[index],
                               os.path.join(self.checkpoint_dir,'in_task_{}_index_{}.png'.format(t, index)))

                with torch.no_grad():
                    logits_adv = self.model(adv_context_images, context_labels, target_images)
                    acc_after = torch.mean(torch.eq(target_labels, torch.argmax(logits_adv, dim=-1)).float()).item()
                    del logits_adv

                    logits = self.model(context_images, context_labels, target_images)
                    acc_before = torch.mean(torch.eq(target_labels, torch.argmax(logits, dim=-1)).float()).item()
                    del logits

                diff = acc_before - acc_after
                print_and_log(self.logfile, "Task = {}, Diff = {}".format(t, diff))

    def attack(self, path, session):
        print_and_log(self.logfile, "")  # add a blank line
        print_and_log(self.logfile, 'Attacking model {0:}: '.format(path))
        self.model = self.init_model()
        self.model.load_state_dict(torch.load(path))
        model_wrapper = Art_Wrapper(self.model)

        classifier = PyTorchClassifier(
            model=model_wrapper,
            clip_values=(-1.0, 1.0),  # Could also get this from context_images
            loss=nn.CrossEntropyLoss(),
            optimizer=self.optimizer,
            input_shape=(1, 3, 84, 84),
            nb_classes=self.args.way,
        )

        # attack = ProjectedGradientDescent(classifier, eps=0.3, eps_step=0.01, max_iter=10)
        attack = FastGradientMethod(classifier)

        for item in self.test_set:

            for t in range(self.args.attack_tasks):

                task_dict = self.dataset.get_test_task(item, session)
                context_images, target_images, context_labels, target_labels, context_images_np = self.prepare_task(task_dict, shuffle=False)
                context_images_attack_all = context_images.clone().detach()

                for c in torch.unique(context_labels):
                    class_index = extract_class_indices(context_labels, c)[0].item()
                    context_x = np.expand_dims(context_images_np[class_index], 0)

                    model_wrapper.init_data(context_images, context_labels, target_images, class_index)

                    adv_x = attack.generate(x=context_x)

                    save_image(adv_x, os.path.join(self.checkpoint_dir, 'adv.png'))
                    save_image(context_x, os.path.join(self.checkpoint_dir, 'in.png'))
                    adv_x_torch = torch.from_numpy(adv_x).to(self.device)
                    context_images_attack_all[class_index] = adv_x_torch

                    with torch.no_grad():
                        logits_adv = self.model(torch.cat([context_images[0:class_index], adv_x_torch,
                            context_images[class_index + 1:]], dim=0), context_labels, target_images)
                        acc_after = torch.mean(torch.eq(target_labels, torch.argmax(logits_adv, dim=-1)).float()).item()

                        logits = self.model(context_images, context_labels, target_images)
                        acc_before = torch.mean(torch.eq(target_labels, torch.argmax(logits, dim=-1)).float()).item()
                        del logits

                    diff = acc_before - acc_after
                    print_and_log(self.logfile, "Task = {}, Class = {} \t Diff = {}".format(t, c, diff))

                print_and_log(self.logfile, "Accuracy before {}".format(acc_after))
                logits = self.model(context_images_attack_all, context_labels, target_images)
                acc_all_attack = torch.mean(torch.eq(target_labels, torch.argmax(logits, dim=-1)).float()).item()
                print_and_log(self.logfile, "Accuracy after {}".format(acc_all_attack))



    def prepare_task(self, task_dict, shuffle=True):
        context_images_np, context_labels_np = task_dict['context_images'], task_dict['context_labels']
        target_images_np, target_labels_np = task_dict['target_images'], task_dict['target_labels']

        context_images_np = context_images_np.transpose([0, 3, 1, 2])

        if shuffle:
            context_images_np, context_labels_np = self.shuffle(context_images_np, context_labels_np)
        context_images = torch.from_numpy(context_images_np)
        context_labels = torch.from_numpy(context_labels_np)

        target_images_np = target_images_np.transpose([0, 3, 1, 2])
        if shuffle:
            target_images_np, target_labels_np = self.shuffle(target_images_np, target_labels_np)
        target_images = torch.from_numpy(target_images_np)
        target_labels = torch.from_numpy(target_labels_np)

        context_images = context_images.to(self.device)
        target_images = target_images.to(self.device)
        context_labels = context_labels.to(self.device)
        target_labels = target_labels.type(torch.LongTensor).to(self.device)

        return context_images, target_images, context_labels, target_labels, context_images_np

    def loss_fn(self, test_logits_sample, test_labels):
        """
        Compute the classification loss.
        """
        size = test_logits_sample.size()
        sample_count = size[0]  # scalar for the loop counter
        num_samples = torch.tensor([sample_count], dtype=torch.float, device=self.device, requires_grad=False)

        log_py = torch.empty(size=(size[0], size[1]), dtype=torch.float, device=self.device)
        for sample in range(sample_count):
            log_py[sample] = -F.cross_entropy(test_logits_sample[sample], test_labels, reduction='none')
        score = torch.logsumexp(log_py, dim=0) - torch.log(num_samples)
        return -torch.sum(score, dim=0)

    def shuffle(self, images, labels):
        """
        Return shuffled data.
        """
        permutation = np.random.permutation(images.shape[0])
        return images[permutation], labels[permutation]

    def use_two_gpus(self):
        use_two_gpus = False
        if self.args.dataset == "meta-dataset":
            if self.args.feature_adaptation == "film+ar" or \
               self.args.batch_normalization == "task_norm-i":
                use_two_gpus = True  # These models do not fit on one GPU, so use model parallelism.

        return use_two_gpus

    def save_checkpoint(self, iteration):
        torch.save({
            'iteration': iteration,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_accuracy': self.validation_accuracies.get_current_best_accuracy_dict(),
        }, os.path.join(self.checkpoint_dir, 'checkpoint.pt'))

    def load_checkpoint(self):
        checkpoint = torch.load(os.path.join(self.checkpoint_dir, 'checkpoint.pt'))
        self.start_iteration = checkpoint['iteration']
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.validation_accuracies.replace(checkpoint['best_accuracy'])

    def register_extra_parameters(self, model):
        for module in model.modules():
            if isinstance(module, TaskNormI):
                module.register_extra_weights()


if __name__ == "__main__":
    main()
