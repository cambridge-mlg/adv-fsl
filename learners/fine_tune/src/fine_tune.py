import torch
import numpy as np
import argparse
import os
from utils import Logger, accuracy
from model import FineTuner
from meta_dataset_reader import MetaDatasetReader, SingleDatasetReader
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Quiet TensorFlow warnings
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)  # Quiet TensorFlow warningsimport globals


def main():
    learner = Learner()
    learner.run()


class Learner:
    def __init__(self):
        self.args = self.parse_command_line()
        self.logger = Logger(self.args.checkpoint_dir, 'log.txt')

        self.logger.print_and_log("Options: %s\n" % self.args)
        self.logger.print_and_log("Checkpoint Directory: %s\n" % self.args.checkpoint_dir)

        gpu_device = 'cuda:0'
        self.device = torch.device(gpu_device if torch.cuda.is_available() else 'cpu')
        self.model = self.init_model()
        self.test_set = self.init_data()
        if self.args.dataset == "meta-dataset":
            self.dataset = MetaDatasetReader(
                data_path=self.args.data_path,
                mode='test',
                train_set=None,
                validation_set=None,
                test_set=self.test_set,
                max_way_train=0,
                max_way_test=50,
                max_support_train=0,
                max_support_test=500,
                max_query_train=0,
                max_query_test=10,
                image_size=self.args.image_size
            )
        else:
            self.dataset = SingleDatasetReader(
                data_path=self.args.data_path,
                mode='test',
                dataset=self.args.dataset,
                way=self.args.way,
                shot=self.args.shot,
                query_train=0,
                query_test=10,
                image_size=self.args.image_size
            )

        self.accuracy_fn = accuracy

    def init_model(self):
        model = FineTuner(args=self.args, device=self.device)
        return model

    def init_data(self):
        if self.args.dataset == "meta-dataset":
            test_set = self.args.test_datasets
        else:
            test_set = [self.args.dataset]

        return test_set

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
        # parser.add_argument('--test_datasets', nargs='+', help='Datasets to use for testing',
        #                     default=["omniglot"])
        parser.add_argument("--data_path", default="../datasets", help="Path to dataset records.")
        parser.add_argument("--pretrained_feature_extractor_path", default="./learners/fine_tune/models/pretrained_mnasnet.pth",
                            help="Path to pretrained feature extractor model.")
        parser.add_argument("--learning_rate", "-lr", type=float, default=0.05, help="Learning rate.")
        parser.add_argument("--weight_decay", "-wd", type=float, default=0.001, help="Weight decay.")
        parser.add_argument("--regularizer_scaling", type=float, default=0.001, help="Scaling for FiLM layer regularization.")
        parser.add_argument("--checkpoint_dir", "-c", default='./checkpoints', help="Directory to save checkpoint to.")
        parser.add_argument("--feature_adaptation", choices=["no_adaptation", "film"], default="film",
                            help="Method to adapt feature extractor parameters.")
        parser.add_argument("--way", type=int, default=5, help="Way of meta-train task.")
        parser.add_argument("--shot", type=int, default=1, help="Shots per class for context.")
        parser.add_argument("--image_size", type=int, default=84, help="Image height and width.")
        parser.add_argument("--iterations", "-i", type=int, default=50, help="Number of fine-tune iterations.")
        parser.add_argument("--test_tasks", "-t", type=int, default=600, help="Number of tasks to test for each dataset.")
        parser.add_argument("--batch_size", "-b", type=int, default=100, help="Batch size.")
        args = parser.parse_args()

        return args

    def run(self):
        config = tf.compat.v1.ConfigProto()
        config.gpu_options.allow_growth = True
        with tf.compat.v1.Session(config=config) as session:
            self.test(session)

    def test(self, session):
        self.logger.print_and_log("")  # add a blank line

        with torch.no_grad():
            for item in self.test_set:
                linear_accuracies = []
                for task in range(self.args.test_tasks):
                    task_dict = self.dataset.get_test_task(item, session)
                    context_images, target_images, context_labels, target_labels = self.prepare_task(task_dict)

                    # fine tune the model to the current task
                    self.model.fine_tune(context_images, context_labels)

                    # test the linear version of the model
                    accuracy = self.model.test_linear(target_images, target_labels)
                    # print("Linear: dataset={}, task={} accuracy={}".format(item, task, accuracy.item()))
                    linear_accuracies.append(accuracy)

                linear_accuracy = np.array(linear_accuracies).mean() * 100.0
                linear_accuracy_confidence = (196.0 * np.array(linear_accuracies).std()) / np.sqrt(len(linear_accuracies))
                self.logger.print_and_log('Linear: {0:}: {1:3.1f}+/-{2:2.1f}'.format(item, linear_accuracy, linear_accuracy_confidence))

    def prepare_task(self, task_dict):
        context_images_np, context_labels_np = task_dict['context_images'], task_dict['context_labels']
        target_images_np, target_labels_np = task_dict['target_images'], task_dict['target_labels']

        context_images_np = context_images_np.transpose([0, 3, 1, 2])
        context_images_np, context_labels_np = self.shuffle(context_images_np, context_labels_np)
        context_images = torch.from_numpy(context_images_np)
        context_labels = torch.from_numpy(context_labels_np)

        target_images_np = target_images_np.transpose([0, 3, 1, 2])
        target_images_np, target_labels_np = self.shuffle(target_images_np, target_labels_np)
        target_images = torch.from_numpy(target_images_np)
        target_labels = torch.from_numpy(target_labels_np)

        context_images = context_images.to(self.device)
        target_images = target_images.to(self.device)
        context_labels = context_labels.type(torch.LongTensor).to(self.device)
        target_labels = target_labels.type(torch.LongTensor).to(self.device)

        return context_images, target_images, context_labels, target_labels

    def shuffle(self, images, labels):
        """
        Return shuffled data.
        """
        permutation = np.random.permutation(images.shape[0])
        return images[permutation], labels[permutation]


if __name__ == "__main__":
    main()
