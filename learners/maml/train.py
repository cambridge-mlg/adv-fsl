import argparse
import torch
import os
import numpy as np

from learners.maml.src.mini_imagenet import MiniImageNetData, prepare_task
from learners.maml.src.omniglot import OmniglotData
from learners.maml.src.maml import MAML
from learners.maml.src.proto_maml import ProtoMAMLv2 as ProtoMAML
from learners.maml.src.shrinkage_maml import SigmaMAML as sMAML
from learners.maml.src.shrinkage_maml import PredCPMAML as pMAML
from learners.maml.src.utils import save_image
from attacks.attack_helpers import create_attack
from attacks.attack_utils import extract_class_indices

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(model, dataset, batch_size):
    """Compute loss and accuracy for a batch of tasks """
    model.set_gradient_steps(args.gradient_steps)
    loss, accuracy = 0, 0
    inputs, outputs = [], []
    for task in range(batch_size):
        task_dict = dataset.get_train_task(way=args.num_classes, 
                                           shot=args.shot, 
                                           target_shot=args.target_shot)
        xc, xt, yc, yt = prepare_task(task_dict)
        inputs.append(xc)

        # Compute task loss
        task_loss, task_accuracy = model.compute_objective(xc, yc, xt, yt,
                                                           accuracy=True)
        loss += task_loss
        accuracy += task_accuracy
    
    # For pMAML, compute prior terms and use as regularization
    if isinstance(model, pMAML) or isinstance(model, sMAML):
        xs = torch.stack(inputs, dim=0)
        prior_terms = model.sigma_regularizers(x=xs)
        loss += torch.stack(prior_terms).sum()

    # Compute average loss across batch
    loss /= batch_size
    accuracy /= batch_size
    return loss, accuracy


def validate(model, dataset, batch_size, test=False):
    """Compute loss and accuracy for a batch of tasks """
    model.set_gradient_steps(test_gradient_steps)
    loss = 0.0
    accuracies = []
    for task in range(batch_size):
        # when testing, target_shot is just shot
        if not test:
            task_dict = dataset.get_validation_task(way=args.num_classes, 
                                                    shot=args.shot, 
                                                    target_shot=args.shot)
        else:
            task_dict = dataset.get_test_task(way=args.num_classes, 
                                              shot=args.shot, 
                                              target_shot=args.shot)
        xc, xt, yc, yt = prepare_task(task_dict)

        # Compute task loss
        task_loss, task_accuracy = model.compute_objective(xc, yc, xt, yt,
                                                           accuracy=True)
        task_loss = task_loss.detach()
        task_accuracy = task_accuracy.detach()
        loss += task_loss
        accuracies.append(task_accuracy.item())

    # Compute average loss across batch
    loss /= batch_size
    accuracy = np.array(accuracies).mean() * 100.0
    confidence = (196.0 * np.array(accuracies).std()) / np.sqrt(len(accuracies))

    return loss, accuracy, confidence


def test(model, data, model_path):
    # load the model
    if device.type == 'cpu':
            load_dict = torch.load(model_path, map_location='cpu')
    else:
            load_dict = torch.load(model_path)
    model.load_state_dict(load_dict)

    # test the  model
    _, test_accuracy, confidence = validate(model, data, batch_size=600,
                                            test=True)
    print("Test Accuracy on {}: {:3.1f}+/-{:2.1f}".format(model_path,
                                                          test_accuracy,
                                                          confidence))


def test_by_example(model, data, model_path):
    # load the model
    if device.type == 'cpu':
            load_dict = torch.load(model_path, map_location='cpu')
    else:
            load_dict = torch.load(model_path)
    model.load_state_dict(load_dict)

    NUM_TEST_TASKS = 600

    model.set_gradient_steps(test_gradient_steps)
    single_task_accuracies = []
    group_task_accuracies = []
    for task in range(NUM_TEST_TASKS):
        # when testing, target_shot is just shot
        task_dict = data.get_test_task(way=args.num_classes,
                                       shot=args.shot,
                                       target_shot=args.shot)
        context_images, target_images, context_labels, target_labels = prepare_task(task_dict)
        # do task one at a time
        num_target_images = (target_images.size())[0]
        single_image_accuracies = []
        for i in range(num_target_images):
            single_target_image = target_images[i:i+1]
            single_target_label = target_labels[i:i+1]
            _, single_image_accuracy = model.compute_objective(context_images, context_labels, single_target_image, single_target_label, accuracy=True)
            single_image_accuracy = single_image_accuracy.detach()
            single_image_accuracies.append(single_image_accuracy.item())
            print('task={0:}/{1:}, image={2:}/{3:}, task accuracy={4:3.1f}, item accuracy={5:3.1f}'
                  .format(task, NUM_TEST_TASKS, i, num_target_images,
                          np.array(single_image_accuracies).mean() * 100.0,
                          np.array(single_task_accuracies).mean() * 100.0 if len(single_task_accuracies) > 0 else 0.0),
                  end='')
            print('\r', end='')
        single_task_accuracy = np.array(single_image_accuracies).mean()
        single_task_accuracies.append(single_task_accuracy)

        _, group_task_accuracy = model.compute_objective(context_images, context_labels, target_images,
                                                           target_labels, accuracy=True)
        group_task_accuracy = group_task_accuracy.detach()
        group_task_accuracies.append(group_task_accuracy.item())

    single_accuracy = np.array(single_task_accuracies).mean() * 100.0
    single_accuracy_confidence = (196.0 * np.array(single_task_accuracies).std()) / np.sqrt(len(single_task_accuracies))
    print('\nSingle: {0:3.1f}+/-{1:2.1f}'.format(single_accuracy, single_accuracy_confidence))
    group_accuracy = np.array(group_task_accuracies).mean() * 100.0
    group_accuracy_confidence = (196.0 * np.array(group_task_accuracies).std()) / np.sqrt(len(group_task_accuracies))
    print('\nGroup: {0:3.1f}+/-{1:2.1f}'.format(group_accuracy, group_accuracy_confidence))


def test_by_class(model, data, model_path):
    # load the model
    if device.type == 'cpu':
        load_dict = torch.load(model_path, map_location='cpu')
    else:
        load_dict = torch.load(model_path)
    model.load_state_dict(load_dict)

    NUM_TEST_TASKS = 600

    model.set_gradient_steps(test_gradient_steps)
    single_task_accuracies = []
    group_task_accuracies = []
    for task in range(NUM_TEST_TASKS):
        # when testing, target_shot is just shot
        task_dict = data.get_test_task(way=args.num_classes,
                                       shot=args.shot,
                                       target_shot=args.shot)
        context_images, target_images, context_labels, target_labels = prepare_task(task_dict)

        # do task one class at a time
        num_target_images = (target_images.size())[0]
        single_class_accuracies = []
        for c in torch.unique(context_labels):
            indices = extract_class_indices(target_labels, c)
            class_target_images = torch.index_select(target_images, 0, indices)
            class_target_labels = torch.index_select(target_labels, 0, indices)

            _, single_class_accuracy = model.compute_objective(context_images, context_labels, class_target_images,
                                                               class_target_labels, accuracy=True)
            single_class_accuracy = single_class_accuracy.detach()
            single_class_accuracies.append(single_class_accuracy.item())

        single_task_accuracy = np.array(single_class_accuracies).mean()
        single_task_accuracies.append(single_task_accuracy)

        _, group_task_accuracy = model.compute_objective(context_images, context_labels, target_images,
                                                         target_labels, accuracy=True)
        group_task_accuracy = group_task_accuracy.detach()
        group_task_accuracies.append(group_task_accuracy.item())

    single_accuracy = np.array(single_task_accuracies).mean() * 100.0
    single_accuracy_confidence = (196.0 * np.array(single_task_accuracies).std()) / np.sqrt(len(single_task_accuracies))
    print('\nSingle: {0:3.1f}+/-{1:2.1f}'.format(single_accuracy, single_accuracy_confidence))
    group_accuracy = np.array(group_task_accuracies).mean() * 100.0
    group_accuracy_confidence = (196.0 * np.array(group_task_accuracies).std()) / np.sqrt(len(group_task_accuracies))
    print('\nGroup: {0:3.1f}+/-{1:2.1f}'.format(group_accuracy, group_accuracy_confidence))


def attack(model, dataset, model_path, tasks, config_path, checkpoint_dir):
    # load the model
    if device.type == 'cpu':
            load_dict = torch.load(model_path, map_location='cpu')
    else:
            load_dict = torch.load(model_path)
    model.load_state_dict(load_dict)

    model.set_gradient_steps(test_gradient_steps)

    attack = create_attack(config_path)

    accuracies_before = []
    accuracies_after = []
    for task in range(tasks):
        # when testing, target_shot is just shot
        task_dict = dataset.get_test_task(way=args.num_classes, shot=args.shot, target_shot=args.shot)
        xc, xt, yc, yt = prepare_task(task_dict)

        if attack.get_attack_mode() == 'context':
            adv_context_images, adv_context_indices = attack.generate(xc, yc, xt, model, model.compute_logits, device)

            for index in adv_context_indices:
                save_image(adv_context_images[index].cpu().detach().numpy(),
                           os.path.join(checkpoint_dir, 'adv_task_{}_index_{}.png'.format(task, index)))
                save_image(xc[index].cpu().detach().numpy(),
                           os.path.join(checkpoint_dir, 'in_task_{}_index_{}.png'.format(task, index)))

            _, acc_after = model.compute_objective(adv_context_images, yc, xt, yt, accuracy=True)

        else:  # target
            adv_target_images = attack.generate(xc, yc, xt, model, model.compute_logits, device)
            for i in range(len(xt)):
                save_image(adv_target_images[i].cpu().detach().numpy(),
                           os.path.join(checkpoint_dir, 'adv_task_{}_index_{}.png'.format(task, i)))
                save_image(xt[i].cpu().detach().numpy(),
                           os.path.join(checkpoint_dir, 'in_task_{}_index_{}.png'.format(task, i)))

            _, acc_after = model.compute_objective(xc, yc, adv_target_images, yt, accuracy=True)

        _, acc_before = model.compute_objective(xc, yc, xt, yt, accuracy=True)

        accuracies_before.append(acc_before.item())
        accuracies_after.append(acc_after.item())

    accuracy = np.array(accuracies_before).mean() * 100.0
    accuracy_confidence = (196.0 * np.array(accuracies_before).std()) / np.sqrt(len(accuracies_before))
    print('Before attack: {0:3.1f}+/-{1:2.1f}'.format(accuracy, accuracy_confidence))

    accuracy = np.array(accuracies_after).mean() * 100.0
    accuracy_confidence = (196.0 * np.array(accuracies_after).std()) / np.sqrt(len(accuracies_after))
    print('After attack: {0:3.1f}+/-{1:2.1f}'.format(accuracy, accuracy_confidence))


# Parse arguments given to the script.
parser = argparse.ArgumentParser()
parser.add_argument('--model',
                    type=str,
                    choices=['maml', 'smaml', 'pmaml', 'proto-maml'],
                    default='maml',
                    help='Choice of model to train')
parser.add_argument('--pi',
                    type=str,
                    choices=['vague', 'cauchy', 'log-cauchy', 'gem', 
                             'exponential'],
                    default='log-cauchy',
                    help='Choice of sigma prior (pMAML and sMAML only)')
parser.add_argument('--beta',
                    type=float,
                    default=0.0001,
                    help='Constant factor in front of prior regularization '\
                    'term (pMAML and sMAML only)')
parser.add_argument('--dataset',
                    type=str,
                    choices=['omniglot', 'mini_imagenet'],
                    default='omniglot',
                    help='Choice of dataset to use')
parser.add_argument('--mode',
                    type=str,
                    choices=['train_and_test', 'test_only', 'attack'],
                    default='train_and_test',
                    help='Whether to run training and testing or just test')
parser.add_argument('--data_path',
                    required=True,
                    type=str,
                    help='Path to mini-imagenet data')
parser.add_argument('--checkpoint_dir',
                    required=True,
                    type=str,
                    help='Path to saved models.')
parser.add_argument('--saved_model',
                    type=str,
                    default='fully_trained',
                    choices=['fully_trained', 'best_validation'],
                    help='Path to model to test.')
parser.add_argument('--num_classes',
                    type=int,
                    default=5,
                    help='"Way" of classification task')
parser.add_argument('--shot',
                    type=int,
                    default=1,
                    help='"Shot" of context set')
parser.add_argument('--target_shot',
                    type=int,
                    default=15,
                    help='"Shot" of target set')
parser.add_argument('--gradient_steps',
                    type=int,
                    default=5,
                    help='Number of inner gradient steps')
parser.add_argument('--iterations',
                    type=int,
                    default=60000,
                    help='Number of training iterations')
parser.add_argument('--tasks_per_batch',
                    type=int,
                    default=4,
                    help='Number of tasks_per_batch')
parser.add_argument('--inner_lr',
                    type=float,
                    default=0.01,
                    help='Inner learning rate')
parser.add_argument("--first_order", dest="first_order",
                    default=False, action="store_true",
                    help="Use first order MAML.")
parser.add_argument('--attack_tasks',
                    type=int,
                    default=10,
                    help='Number of attack tasks.')
parser.add_argument("--attack_config_path",
                    help="Path to attack config file in yaml format.")
parser.add_argument("--test_model_path",
                    help="Path to model to be tested.")
parser.add_argument("--attack_model_path",
                    help="Path to model to attack.")
parser.add_argument("--test_by_example", dest="test_by_example",
                    default=False, action="store_true",
                    help="Test one example at a time.")
parser.add_argument("--test_by_class", dest="test_by_class",
                    default=False, action="store_true",
                    help="Test one class at a time.")
args = parser.parse_args()

# Create checkpoint directory (if required)
if not os.path.exists(args.checkpoint_dir):
    os.makedirs(args.checkpoint_dir)

# Set model path
model_str = args.saved_model + '.pt'
args.model_path = os.path.join(args.checkpoint_dir, model_str)

# Initialize data generator
if args.dataset == 'mini_imagenet':
    data = MiniImageNetData(path=args.data_path, seed=111)
    in_channels = 3
    num_channels = 32
    max_pool = True
    flatten = True
    hidden_dim = 800
    test_gradient_steps = 10
else:
    data = OmniglotData(path=args.data_path, seed=111)
    in_channels = 1
    num_channels = 64
    max_pool = False
    flatten = False
    hidden_dim = 64
    if args.num_classes == 5:
        test_gradient_steps = 3
    else:
        test_gradient_steps = 5

# Initialize model
if args.model in ['maml', 'proto-maml']:
    maml = MAML if args.model == 'maml' else ProtoMAML
    model = maml(update_lr=args.inner_lr,
                 in_channels=in_channels,
                 num_channels=num_channels,
                 num_classes=args.num_classes,
                 gradient_steps=args.gradient_steps,
                 max_pool=max_pool,
                 hidden_dim=hidden_dim,
                 flatten=flatten,
                 first_order=args.first_order)

elif args.model in ['smaml', 'pmaml']:
    maml = sMAML if args.model == 'smaml' else pMAML
    model = maml(update_lr=args.inner_lr,
                 in_channels=in_channels,
                 num_channels=num_channels,
                 num_classes=args.num_classes,
                 pi=args.pi,
                 beta=args.beta,
                 gradient_steps=args.gradient_steps,
                 max_pool=max_pool,
                 hidden_dim=hidden_dim,
                 flatten=flatten)

else:
    raise ValueError('Invalid model choice')

model.to(device)
model.train()

if args.mode == 'train_and_test':
    # Initialize outer-loop optimizer
    optimizer = torch.optim.Adam(model.parameters(), 0.001)

    # Some training constants
    PRINT_FREQUENCY = 500
    VALIDATE_FREQUENCY = 2000

    # create checkpoint directory if it does not already exist
    if not os.path.exists(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir)

    # Training loop
    best_validation_accuracy = 0.0
    for iteration in range(args.iterations):

        loss, accuracy = train(model, data, args.tasks_per_batch)
        # Compute gradient to global parameters and take step
        optimizer.zero_grad()
        loss.backward()

        if torch.isinf(loss).item():
            print("Loss is inf, avoiding optimization step")
        else:
            optimizer.step()

        if (iteration + 1) % PRINT_FREQUENCY == 0:
            # print training stats
            print('Task [{}/{}], Train Loss: {:.7f}, '\
                  'Train Acc: {:.7f}'.format(iteration + 1,
                                             args.iterations,
                                             loss.item(),
                                             accuracy.item()))

        if (iteration + 1) % VALIDATE_FREQUENCY == 0:
            val_loss, val_accuracy, confidence = validate(model, data,
                                                          batch_size=128)

            print('Validating at [{}/{}], Validation Loss: {:.7f}, '\
                  'Validation Acc: {:3.1f}+/-{:2.1f}'.format(iteration + 1,
                                                  args.iterations,
                                                  val_loss.item(),
                                                  val_accuracy, confidence))

            if isinstance(model, pMAML) or isinstance(model, sMAML):
                model.print_sigmas()

            # save model if validation accuracy is the best so far
            if val_accuracy > best_validation_accuracy:
                best_validation_accuracy = val_accuracy
                torch.save(model.state_dict(), os.path.join(
                    args.checkpoint_dir, 'best_validation.pt'))
                print("Best validation model updated.")

    # save the fully trained model
    torch.save(model.state_dict(), os.path.join(args.checkpoint_dir,
                                                'fully_trained.pt'))

    # After training -- test the model on test set
    test(model, data, os.path.join(args.checkpoint_dir, 'fully_trained.pt'))
    test(model, data, os.path.join(args.checkpoint_dir, 'best_validation.pt'))

elif args.mode == 'attack':
    attack(model, data, args.attack_model_path, args.attack_tasks, args.attack_config_path, args.checkpoint_dir)

else:  # test only
    if args.test_by_example:
        test_by_example(model, data, args.test_model_path)
    elif args.test_by_class:
        test_by_class(model, data, args.test_model_path)
    else:  # test normally
        # Test fully trained
        args.model_path = os.path.join(args.checkpoint_dir, 'fully_trained.pt')
        test(model, data, args.model_path)
        # Test best validation model
        args.model_path = os.path.join(args.checkpoint_dir, 'best_validation.pt')
        test(model, data, args.model_path)
