import torch
import numpy as np
import argparse
import os
from utils import Logger, accuracy
from model import FineTuner
from meta_dataset_reader import MetaDatasetReader, SingleDatasetReader
from attacks.attack_utils import AdversarialDataset
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Quiet TensorFlow warnings
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)  # Quiet TensorFlow warningsimport globals


def main():
    learner = Learner()
    learner.run()


class Learner:
    def __init__(self):
        self.args = self.parse_command_line()
        self.logger = Logger(self.args.checkpoint_dir, self.args.log_file)

        self.logger.print_and_log("Options: %s\n" % self.args)
        self.logger.print_and_log("Checkpoint Directory: %s\n" % self.args.checkpoint_dir)

        gpu_device = 'cuda:0'
        self.device = torch.device(gpu_device if torch.cuda.is_available() else 'cpu')
        self.model = self.init_model()
        self.dataset = AdversarialDataset(self.args.data_path)

        self.max_test_tasks = min(self.dataset.get_num_tasks(), self.args.test_tasks)

        self.accuracy_fn = accuracy

    def init_model(self):
        model = FineTuner(args=self.args, device=self.device)
        return model


    """
    Command line parser
    """
    def parse_command_line(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("--data_path", default="../datasets", help="Path to dataset records.")
        parser.add_argument("--feature_extractor", choices=["mnasnet", "resnet"], default="mnasnet",
                            help="Dataset to use.")
        parser.add_argument("--pretrained_feature_extractor_path", default="./learners/fine_tune/models/pretrained_mnasnet.pth",
                            help="Path to pretrained feature extractor model.")
        parser.add_argument("--learning_rate", "-lr", type=float, default=0.05, help="Learning rate.")
        parser.add_argument("--weight_decay", "-wd", type=float, default=0.001, help="Weight decay.")
        parser.add_argument("--regularizer_scaling", type=float, default=0.001, help="Scaling for FiLM layer regularization.")
        parser.add_argument("--checkpoint_dir", "-c", default='./checkpoints', help="Directory to save checkpoint to.")
        parser.add_argument("--feature_adaptation", choices=["no_adaptation", "film"], default="film",
                            help="Method to adapt feature extractor parameters.")
        parser.add_argument("--iterations", "-i", type=int, default=50, help="Number of fine-tune iterations.")
        parser.add_argument("--test_tasks", "-t", type=int, default=1000, help="Number of tasks to test for each dataset.")
        parser.add_argument("--batch_size", "-b", type=int, default=1000, help="Batch size.")
        parser.add_argument("--log_file", default="log.tx", help="Name of log file")
        args = parser.parse_args()

        return args

    def run(self):
        config = tf.compat.v1.ConfigProto()
        config.gpu_options.allow_growth = True
        with tf.compat.v1.Session(config=config) as session:
            self.finetune(session)

    def eval(self, task_index):
        eval_acc = []
        target_image_sets, target_label_sets = self.dataset.get_eval_task(task_index)
        for s in range(len(target_image_sets)):
            accuracy = self.model.test_linear(target_image_sets[s], target_label_sets[s])
            eval_acc.append(accuracy)
        return np.array(eval_acc).mean()

    def print_average_accuracy(self, accuracies, descriptor):
        accuracy = np.array(accuracies).mean() * 100.0
        accuracy_confidence = (196.0 * np.array(accuracies).std()) / np.sqrt(len(accuracies))
        self.logger.print_and_log('{0:} {1:3.1f}+/-{2:2.1f}'.format(descriptor, accuracy, accuracy_confidence))

    def finetune(self, session):
        self.logger.print_and_log("")  # add a blank line

        with torch.no_grad():
            clean_acc_0 = []
            adv_acc_0 = []
            clean_acc = []
            adv_acc = []

            for task in range(self.max_test_tasks):
                # Clean task
                context_images, context_labels, target_images, target_labels = self.dataset.get_clean_task(task)
                # fine tune the model to the current task
                self.model.fine_tune(context_images, context_labels)
                accuracy = self.model.test_linear(target_images, target_labels)
                clean_acc_0.append(accuracy)

                clean_acc.append(self.eval(task))

                # Adversarial task
                adv_images, context_labels, target_images, target_labels = self.dataset.get_adversarial_task(task)
                # fine tune the model to the current task
                self.model.fine_tune(adv_images, context_labels)
                accuracy = self.model.test_linear(target_images, target_labels)
                adv_acc_0.append(accuracy)

                adv_acc.append(self.eval(task))

            self.print_average_accuracy(clean_acc_0, "Clean Acc (gen setting)")
            self.print_average_accuracy(clean_acc, "Clean Acc")
            self.print_average_accuracy(adv_acc_0, "Adv Acc (gen setting)")
            self.print_average_accuracy(adv_acc, "Adv Acc")

            #linear_accuracy = np.array(linear_accuracies).mean() * 100.0
            #linear_accuracy_confidence = (196.0 * np.array(linear_accuracies).std()) / np.sqrt(len(linear_accuracies))
            #self.logger.print_and_log('Linear: {0:}: {1:3.1f}+/-{2:2.1f}'.format(item, linear_accuracy, linear_accuracy_confidence))

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
