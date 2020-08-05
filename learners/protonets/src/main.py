import torch
import numpy as np
import argparse
import os
from learners.protonets.src.utils import print_and_log, get_log_files, categorical_accuracy, loss, get_labels
from learners.protonets.src.model import ProtoNets
from learners.protonets.src.data import MiniImageNetData, OmniglotData
from attacks.attack_helpers import create_attack
from attacks.attack_utils import save_image, split_target_set, extract_class_indices
from matplotlib import pyplot as plt
import pickle
import torch.nn.functional as F
import matplotlib.pylab as pl
from matplotlib.colors import ListedColormap

NUM_VALIDATION_TASKS = 400
NUM_TEST_TASKS = 1000
PRINT_FREQUENCY = 100


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
        self.loss = loss
        if self.args.dataset == "mini_imagenet":
            self.dataset = MiniImageNetData(self.args.data_path, 111)
        elif self.args.dataset == "omniglot":
            self.dataset = OmniglotData(self.args.data_path, 111)
        else:
            self.dataset = None

        self.accuracy_fn = categorical_accuracy
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        self.start_iteration = 0
        if self.args.resume_from_checkpoint:
            self.load_checkpoint()
        self.optimizer.zero_grad()
        self.best_validation_accuracy = 0.0

    def init_model(self):
        model = ProtoNets(args=self.args).to(self.device)
        model.train()
        return model

    """
    Command line parser
    """
    def parse_command_line(self):
        parser = argparse.ArgumentParser()

        parser.add_argument("--dataset", choices=['omniglot', "mini_imagenet"], default="mini_imagenet", help="Dataset to use.")
        parser.add_argument("--data_path", default="../datasets", help="Path to dataset records.")
        parser.add_argument("--mode", choices=["train", "test", "train_test", "attack"], default="train_test",
                            help="Whether to run training only, testing only, or both training and testing.")
        parser.add_argument("--attack_config_path", help="Path to attack config file in yaml format.")
        parser.add_argument("--attack_tasks", "-a", type=int, default=10, help="Number of tasks when performing attack.")
        parser.add_argument("--learning_rate", "-lr", type=float, default=0.001, help="Learning rate.")
        parser.add_argument("--tasks_per_batch", type=int, default=1,
                            help="Number of tasks between parameter optimizations.")
        parser.add_argument("--checkpoint_dir", "-c", default='../checkpoints', help="Directory to save checkpoint to.")
        parser.add_argument("--test_model_path", "-m", default=None, help="Path to model to load and test.")
        parser.add_argument("--training_iterations", "-i", type=int, default=20000, help="Number of meta-training iterations.")
        parser.add_argument("--val_freq", type=int, default=200, help="Number of iterations between validations.")
        parser.add_argument("--resume_from_checkpoint", "-r", dest="resume_from_checkpoint", default=False,
                            action="store_true", help="Restart from latest checkpoint.")
        parser.add_argument("--train_way", type=int, default=20, help="Way of meta-train task.")
        parser.add_argument("--train_shot", type=int, default=5, help="Shots per class for meta-train context sets.")
        parser.add_argument("--test_way", type=int, default=5, help="Way of meta-test task.")
        parser.add_argument("--test_shot", type=int, default=5, help="Shots per class for meta-test context sets.")
        parser.add_argument("--query", type=int, default=15, help="Shots per class for target")
        parser.add_argument("--swap_attack", default=False,
                            help="When attacking, should the attack be a swap attack or not.")
        parser.add_argument("--bottleneck", dest="bottleneck", default=False, action="store_true",
                            help="Use the 2D bottleneck feature extractor for analysis.")
        parser.add_argument("--save_samples", default=False,
                            help="Output samples of the clean and adversarial images")
        args = parser.parse_args()

        return args

    def run(self):
        if self.args.mode == 'train' or self.args.mode == 'train_test':
            train_accuracies = []
            losses = []
            total_iterations = self.args.training_iterations
            for iteration in range(self.start_iteration, total_iterations):
                current_lr = self.adjust_learning_rate(iteration)
                # torch.set_grad_enabled(True)
                task_dict = self.dataset.get_train_task(self.args.train_way,
                                                        self.args.train_shot,
                                                        self.args.query)
                task_loss, task_accuracy = self.train_task(task_dict)
                losses.append(task_loss)
                train_accuracies.append(task_accuracy)

                # optimize
                if ((iteration + 1) % self.args.tasks_per_batch == 0) or (iteration == (total_iterations - 1)):
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                if (iteration + 1) % PRINT_FREQUENCY == 0:
                    # print training stats
                    print_and_log(self.logfile,'Task [{}/{}], Train Loss: {:.7f}, Train Accuracy: {:.7f}, lr: {:.7f}'
                                  .format(iteration + 1, total_iterations, torch.Tensor(losses).mean().item(),
                                          torch.Tensor(train_accuracies).mean().item(), current_lr))
                    train_accuracies = []
                    losses = []

                if ((iteration + 1) % self.args.val_freq == 0) and (iteration + 1) != total_iterations:
                    # validate
                    accuracy = self.validate()
                    # save the model if validation is the best so far
                    if accuracy > self.best_validation_accuracy:
                        self.best_validation_accuracy = accuracy
                        torch.save(self.model.state_dict(), self.checkpoint_path_validation)
                        print_and_log(self.logfile, 'Best validation model was updated.')
                        print_and_log(self.logfile, '')
                    self.save_checkpoint(iteration + 1)

            # save the final model
            torch.save(self.model.state_dict(), self.checkpoint_path_final)

        if self.args.mode == 'train_test':
            self.test(self.checkpoint_path_final)
            self.test(self.checkpoint_path_validation)

        if self.args.mode == 'test':
            self.test(self.args.test_model_path)

        if self.args.mode == 'attack':
            if self.args.swap_attack and not self.args.bottleneck:
                self.attack_swap(self.args.test_model_path)
            elif not self.args.swap_attack and self.args.bottleneck:
                self.plot_attacks(self.args.test_model_path)
            elif not self.args.swap_attack and not self.args.bottleneck:
                self.attack(self.args.test_model_path)
            else:
                print("Invalid command line parameters for attack. Attack must be either a bottleneck, or a swap, or a normal attack.")

        self.logfile.close()

    def train_task(self, task_dict):
        self.model.train()
        context_images, target_images, context_labels, target_labels = self.prepare_task(task_dict, shuffle=False)

        logits = self.model(context_images, context_labels, target_images)
        # target_labels = get_labels(self.args.train_way, self.args.query)
        task_loss = self.loss(logits, target_labels) / self.args.tasks_per_batch
        task_loss.backward()
        accuracy = self.accuracy_fn(logits, target_labels)
        return task_loss, accuracy

    def validate(self):
        self.model.eval()
        with torch.no_grad():
            accuracies = []
            for _ in range(NUM_VALIDATION_TASKS):
                task_dict = self.dataset.get_validation_task(self.args.test_way, self.args.test_shot, self.args.query)
                context_images, target_images, context_labels, target_labels = self.prepare_task(task_dict, shuffle=False)
                logits = self.model(context_images, context_labels, target_images)
                accuracy = self.accuracy_fn(logits, target_labels)
                accuracies.append(accuracy)
                del logits

            accuracy = np.array(accuracies).mean() * 100.0
            confidence = (196.0 * np.array(accuracies).std()) / np.sqrt(len(accuracies))
            print_and_log(self.logfile, 'Validation Accuracy: {0:3.1f}+/-{1:2.1f}'.format(accuracy, confidence))

        return accuracy

    def test(self, path):
        print_and_log(self.logfile, "")  # add a blank line
        print_and_log(self.logfile, 'Testing model {0:}: '.format(path))
        self.model = self.init_model()
        self.model.load_state_dict(torch.load(path))

        self.model.eval()
        with torch.no_grad():
            accuracies = []
            for _ in range(NUM_TEST_TASKS):
                task_dict = self.dataset.get_test_task(self.args.test_way,
                                                       self.args.test_shot,
                                                       self.args.query)
                context_images, target_images, context_labels, target_labels = self.prepare_task(task_dict, shuffle=False)
                logits = self.model(context_images, context_labels, target_images)
                accuracy = self.accuracy_fn(logits, target_labels)
                accuracies.append(accuracy)
                del logits

            accuracy = np.array(accuracies).mean() * 100.0
            accuracy_confidence = (196.0 * np.array(accuracies).std()) / np.sqrt(len(accuracies))

            print_and_log(self.logfile, 'Test Accuracy: {0:3.1f}+/-{1:2.1f}'.format(accuracy, accuracy_confidence))

    def calc_accuracy(self, context_images, context_labels, target_images, target_labels):
        logits = self.model(context_images, context_labels, target_images)
        acc = torch.mean(torch.eq(target_labels.long(), torch.argmax(logits, dim=-1).long()).float()).item()
        del logits
        return acc

    def print_average_accuracy(self, accuracies, descriptor):
        accuracy = np.array(accuracies).mean() * 100.0
        accuracy_confidence = (196.0 * np.array(accuracies).std()) / np.sqrt(len(accuracies))
        print_and_log(self.logfile,
                      '{0:} : {1:3.1f}+/-{2:2.1f}'.format(descriptor, accuracy, accuracy_confidence))

    def save_image_pair(self, adv_img, clean_img, task_no, index):
        save_image(adv_img.cpu().detach().numpy(),
                   os.path.join(self.checkpoint_dir, 'adv_task_{}_index_{}.png'.format(task_no, index)),
                   scaling='zero_to_one')
        save_image(clean_img.cpu().detach().numpy(), os.path.join(self.checkpoint_dir,
                                                                  'in_task_{}_index_{}.png'.format(task_no, index)),
                   scaling='zero_to_one')

    def attack_swap(self, path):
        print_and_log(self.logfile, "")  # add a blank line
        print_and_log(self.logfile, 'Swap attacking model {0:}: '.format(path))
        self.model = self.init_model()
        self.model.load_state_dict(torch.load(path))
        self.model.eval()

        context_attack = create_attack(self.args.attack_config_path, self.checkpoint_dir)
        context_attack.set_attack_mode('context')
        assert context_attack.get_shot_fraction() == 1.0
        assert context_attack.get_class_fraction() == 1.0

        target_attack = create_attack(self.args.attack_config_path, self.checkpoint_dir)
        target_attack.set_attack_mode('target')

        # Accuracies for setting in which we generate attacks.
        # Useful for debugging attacks
        gen_clean_accuracies = []
        gen_adv_context_accuracies = []
        gen_adv_target_accuracies = []

        # Accuracies for evaluation setting
        clean_accuracies = []
        clean_target_as_context_accuracies = []
        adv_context_accuracies = []
        adv_target_accuracies = []
        adv_target_as_context_accuracies = []
        adv_context_as_target_accuracies = []

        for t in range(self.args.attack_tasks):
            task_dict = self.dataset.get_test_task(self.args.test_way, self.args.test_shot, self.args.query)
            context_images, all_target_images, context_labels, all_target_labels = self.prepare_task(task_dict, shuffle=False)

            # Select as many target images as context images to be used on the attack
            # The rest will be used for evaluation
            assert context_images.shape[0] <= all_target_images.shape[0]
            split_target_images, split_target_labels = split_target_set(all_target_images, all_target_labels, self.args.test_shot)
            target_images = split_target_images[0]
            target_labels = split_target_labels[0]

            adv_context_images, adv_context_indices = context_attack.generate(
                context_images, context_labels, target_images, target_labels,
                self.model, self.model, self.device)

            adv_target_images, adv_target_indices = target_attack.generate(
                context_images, context_labels, target_images, target_labels,
                self.model, self.model, self.device)

            assert [x.item() for x in adv_context_indices] == adv_target_indices

            if self.args.save_samples and t < 10:
                for index in adv_context_indices:
                    self.save_image_pair(adv_context_images[index], context_images[index], t, index)
                    self.save_image_pair(adv_target_images[index], target_images[index], t, index)

            with torch.no_grad():
                # Evaluate in normal/generation setting
                gen_clean_accuracies.append(
                    self.calc_accuracy(context_images, context_labels, target_images, target_labels))
                gen_adv_context_accuracies.append(
                    self.calc_accuracy(adv_context_images, context_labels, target_images, target_labels))
                gen_adv_target_accuracies.append(
                    self.calc_accuracy(context_images, context_labels, adv_target_images, target_labels))

                # Evaluate on independent target sets
                for s in range(1, len(split_target_images)):
                    clean_accuracies.append(self.calc_accuracy(context_images, context_labels, split_target_images[s],
                                                               split_target_labels[s]))
                    clean_target_as_context_accuracies.append(
                        self.calc_accuracy(target_images, target_labels, split_target_images[s],
                                           split_target_labels[s]))

                    adv_context_accuracies.append(
                        self.calc_accuracy(adv_context_images, context_labels, split_target_images[s],
                                           split_target_labels[s]))
                    adv_target_accuracies.append(
                        self.calc_accuracy(split_target_images[s], split_target_labels[s], adv_target_images,
                                           target_labels))

                    adv_target_as_context_accuracies.append(
                        self.calc_accuracy(adv_target_images, target_labels, split_target_images[s],
                                           split_target_labels[s]))
                    adv_context_as_target_accuracies.append(
                        self.calc_accuracy(split_target_images[s], split_target_labels[s], adv_context_images,
                                           context_labels))

                del adv_context_images, adv_target_images

        self.print_average_accuracy(gen_clean_accuracies, "Gen setting: Clean accuracy")
        self.print_average_accuracy(gen_adv_context_accuracies, "Gen setting: Context attack accuracy")
        self.print_average_accuracy(gen_adv_target_accuracies, "Gen setting: Target attack accuracy")

        self.print_average_accuracy(clean_accuracies, "Clean accuracy")
        self.print_average_accuracy(clean_target_as_context_accuracies, "Clean Target as Context accuracy")
        self.print_average_accuracy(adv_context_accuracies, "Context attack accuracy")
        self.print_average_accuracy(adv_target_as_context_accuracies, "Adv Target as Context accuracy")
        self.print_average_accuracy(adv_target_accuracies, "Target attack accuracy")
        self.print_average_accuracy(adv_context_as_target_accuracies, "Adv Context as Target")

    def attack(self, path):
        print_and_log(self.logfile, "")  # add a blank line
        print_and_log(self.logfile, 'Attacking model {0:}: '.format(path))
        self.model = self.init_model()
        self.model.load_state_dict(torch.load(path))
        self.model.eval()

        attack = create_attack(self.args.attack_config_path, self.checkpoint_dir)

        accuracies_before = []
        accuracies_after = []
        for t in range(self.args.attack_tasks):
            task_dict = self.dataset.get_test_task(self.args.test_way, self.args.test_shot, self.args.query)
            context_images, target_images, context_labels, target_labels = self.prepare_task(task_dict, shuffle=False)

            if attack.get_attack_mode() == 'context':
                adv_context_images, adv_context_indices = attack.generate(context_images, context_labels, target_images,
                                                                target_labels, self.model, self.model, self.device)

                if self.args.save_samples and t < 10:
                    for index in adv_context_indices:
                        self.save_image_pair(adv_context_images[index], context_images[index], t, index)

                with torch.no_grad():
                    acc_after = self.calc_accuracy(adv_context_images, context_labels, target_images, target_labels)

            else:  # target
                adv_target_images, _ = attack.generate(context_images, context_labels, target_images, target_labels,
                                                    self.model, self.model, self.device)
                if self.args.save_samples and t < 10:
                    for i in range(len(target_images)):
                        self.save_image_pair(adv_target_images[index], target_images[index], t, i)

                with torch.no_grad():
                    acc_after = self.calc_accuracy(context_images, context_labels, adv_target_images, target_labels)

            acc_before = self.calc_accuracy(context_images, context_labels, target_images, target_labels)

            accuracies_before.append(acc_before)
            accuracies_after.append(acc_after)

        self.print_average_accuracy(accuracies_before, "Before attack:")
        self.print_average_accuracy(accuracies_after, "After attack:")

    def plot_attacks(self, path):
        print_and_log(self.logfile, "")  # add a blank line
        print_and_log(self.logfile, 'Attacking model {0:}: '.format(path))
        self.model = self.init_model()
        self.model.load_state_dict(torch.load(path))
        self.model.eval()

        attack = create_attack(self.args.attack_config_path, self.checkpoint_dir)
        attack.set_verbose(True)

        accuracies_before = []
        accuracies_after = []
        for t in range(self.args.attack_tasks):
            task_dict = self.dataset.get_test_task(self.args.test_way, self.args.test_shot, self.args.query)
            context_images, target_images, context_labels, target_labels = self.prepare_task(task_dict, shuffle=False)

            if attack.get_attack_mode() == 'context':
                adv_context_images, adv_context_indices, extra_info = attack.generate(context_images, context_labels, target_images,
                                                                target_labels, self.model, self.model, self.device)

                if self.args.save_samples and t < 10:
                    for index in adv_context_indices:
                        self.save_image_pair(adv_context_images[index], context_images[index], t, index)

                with torch.no_grad():
                    acc_after = self.calc_accuracy(adv_context_images, context_labels, target_images, target_labels)

            else:  # target
                adv_target_images, _, extra_info = attack.generate(context_images, context_labels, target_images, target_labels,
                                                    self.model, self.model, self.device)
                if self.args.save_samples and t < 10:
                    for i in range(len(target_images)):
                        self.save_image_pair(adv_target_images[i], target_images[i], t, i)

                with torch.no_grad():
                    acc_after = self.calc_accuracy(context_images, context_labels, adv_target_images, target_labels)

            acc_before = self.calc_accuracy(context_images, context_labels, target_images, target_labels)

            accuracies_before.append(acc_before)
            accuracies_after.append(acc_after)

            intermediate_attack_imgs = extra_info['adv_images']
            intermediate_logits = extra_info['target_logits']
            classes = torch.unique(context_labels)
            clean_colors = ['navy', 'darkred', 'darkgreen', 'gold', 'darkviolet', 'c', 'm']
            colors = ['b', 'r', 'g', 'orange', 'purple', 'cyan', 'm']
            color_maps = []
            edge_colors = ['k', 'k', 'k', 'k', 'k', 'k', 'k']
            markers = ['v', '^', '<', '>', 'd']
            target_markers = ['1', '2', '3', '4', '+']
            assert len(classes) <= len(colors)


            #Make custom colormaps with the transparency we need
            #No, I don't know how I got here
            def_color_maps = [pl.cm.Blues, pl.cm.Reds, pl.cm.Greens, pl.cm.Oranges, pl.cm.Purples]
            # Get the colormap colors
            for cmap in def_color_maps:
                my_cmap = cmap(np.arange(cmap.N))
                # Set alpha
                my_cmap[:,-1] = np.linspace(0, 1, cmap.N)
                # Create new colormap
                my_cmap = ListedColormap(my_cmap)
                color_maps.append(my_cmap)




            with torch.no_grad():
                min, max = np.Inf, -np.Inf
                clean_context_features = self.model.feature_extractor(context_images).cpu()
                target_features = self.model.feature_extractor(target_images).cpu()
                num_classes = len(classes)
                #out_name = os.path.join(self.checkpoint_dir, "task_{}_pgd_attack_{}_clean_zoom.png".format(t, attack.attack_mode))
                #self._plot_decision_regions(clean_context_features, target_features, num_classes, color_maps, markers, colors, edge_colors, context_labels, out_name, sym_bound=1)
                #out_name = os.path.join(self.checkpoint_dir, "task_{}_pgd_attack_{}_clean.png".format(t, attack.attack_mode))
                #self._plot_decision_regions(clean_context_features, target_features, num_classes, color_maps, markers, colors, edge_colors, context_labels, out_name)

                # self._quick_dump(clean_context_features, os.path.join(self.checkpoint_dir, 'clean_features.pickle'))
                for k in range(0, len(intermediate_attack_imgs)):
                    context_features = self.model.feature_extractor(intermediate_attack_imgs[k]).cpu()
                    # self._quick_dump(context_features, os.path.join(self.checkpoint_dir, 'adv_features_{:02d}.pickle'.format(k)))
                    out_name = os.path.join(self.checkpoint_dir, "task_{}_pgd_attack_{}_{}_{}_iter_{:03d}.png".format(t, attack.attack_mode, attack.target_loss_mode, attack.targeted, k))
                    self._plot_decision_regions(clean_context_features, target_features, num_classes, color_maps,
                                                markers, colors, edge_colors, context_labels, out_name, context_features)
            # print("Min = {}, max = {}".format(min, max))

        self.print_average_accuracy(accuracies_before, "Before attack:")
        self.print_average_accuracy(accuracies_after, "After attack:")

    def _plot_decision_regions(self, clean_context_features, target_features, num_classes, color_maps, markers, colors, edge_colors, context_labels, out_name, context_features=None, sym_bound=5):
        resolution = 50
        xx, yy = np.meshgrid(
            np.linspace(-sym_bound, sym_bound, resolution),  # np.geomspace(-10, 10, resolution)
            np.linspace(-sym_bound, sym_bound, resolution)
        )
        grid_points = np.c_[xx.ravel(), yy.ravel()]
        grid_tensor = torch.from_numpy(grid_points).type(torch.DoubleTensor).to(self.device)

        # If we have context_features, we should use them
        if context_features is not None:
            grid_logits = self.model.forward_embeddings(context_features.type(torch.DoubleTensor).to(self.device),
                                                    grid_tensor).cpu()
        else:
            grid_logits = self.model.forward_embeddings(clean_context_features.type(torch.DoubleTensor).to(self.device),
                                                    grid_tensor).cpu()

        # normalize within each row
        grid_pred = torch.argmax(grid_logits, dim=-1)
        grid_conf = torch.max(F.softmax(grid_logits, dim=1), dim=1)[0]

        fig, ax = plt.subplots(1, 1)
        fig.set_size_inches(16, 16)
        plt.ylim(-sym_bound, sym_bound)
        plt.xlim(-sym_bound, sym_bound)

        for c in range(num_classes):
            grid_c = (grid_pred == c).type(torch.DoubleTensor)
            height = (grid_c * grid_conf).reshape(resolution, resolution)
            ax.contourf(xx, yy, height, cmap=color_maps[c])

        for c in range(num_classes):
            shot_indices = extract_class_indices(context_labels, c)
            # Get the mean of the embeddings for this class
            centroid = torch.zeros_like(clean_context_features[0])
            clean_centroid = torch.zeros_like(clean_context_features[0])
            for j, i in enumerate(shot_indices):
                centroid = centroid + context_features[i]
                clean_centroid = clean_centroid + clean_context_features[i]
                #plt.scatter(clean_context_features[i, 0], clean_context_features[i, 1], marker='.', c=colors[c],
                #            edgecolors=edge_colors[c], s=150)
                plt.scatter(target_features[i, 0], target_features[i, 1], marker='1', c=colors[c],
                            alpha=0.8, edgecolors=edge_colors[c], s=150)
                if context_features is not None:
                    plt.scatter(context_features[i, 0], context_features[i, 1], marker=markers[j], c=colors[c],
                            edgecolors=edge_colors[c], s=150, alpha=0.8)
            centroid = centroid/float(len(shot_indices))
            clean_centroid = clean_centroid/float(len(shot_indices))
            if len(shot_indices) > 1:
                plt.scatter(centroid[0], centroid[1], marker='s', c=colors[c], edgecolors=edge_colors[c], s=250, alpha=0.8)
                plt.scatter(clean_centroid[0], clean_centroid[1], marker='o', c=colors[c], edgecolors=edge_colors[c], s=250, alpha=0.8)


        # ax.scatter(grid_points[:, 0], grid_points[:, 1], c=grid_pred, s=grid_conf*100) #, s=grid_conf*10
        plt.savefig(out_name)
        plt.close()

    def _quick_dump(self, val, name):
        fout = open(name, "wb")
        pickle.dump(val, fout)
        fout.close()

    def prepare_task(self, task_dict, shuffle):
        context_images, context_labels = task_dict['context_images'], task_dict['context_labels']
        target_images, target_labels = task_dict['target_images'], task_dict['target_labels']

        if shuffle:
            context_images, context_labels = self.shuffle(context_images, context_labels)
            target_images, target_labels = self.shuffle(target_images, target_labels)

        context_images = context_images.to(self.device)
        target_images = target_images.to(self.device)
        context_labels = torch.from_numpy(context_labels).to(self.device)
        target_labels = torch.from_numpy(target_labels).type(torch.LongTensor).to(self.device)

        return context_images, target_images, context_labels, target_labels

    def shuffle(self, images, labels):
        """
        Return shuffled data.
        """
        permutation = np.random.permutation(images.shape[0])
        return images[permutation], labels[permutation]

    def save_checkpoint(self, iteration):
        torch.save({
            'iteration': iteration,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_accuracy': self.best_validation_accuracy,
        }, os.path.join(self.checkpoint_dir, 'checkpoint.pt'))

    def load_checkpoint(self):
        checkpoint = torch.load(os.path.join(self.checkpoint_dir, 'checkpoint.pt'))
        self.start_iteration = checkpoint['iteration']
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.best_validation_accuracy = checkpoint['best_accuracy']

    def adjust_learning_rate(self, iteration):
        """
        Sets the learning rate to the initial LR decayed by 2 every 2000 tasks
        """
        lr = self.args.learning_rate * (0.5 ** (iteration // 2000))
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

        return lr


if __name__ == "__main__":
    main()
