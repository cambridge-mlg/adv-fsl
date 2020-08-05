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
from attacks.attack_utils import extract_class_indices, Logger, split_target_set

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
    logger.print_and_log("Test Accuracy on {}: {:3.1f}+/-{:2.1f}".format(model_path,
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
            logger.print_and_log('task={0:}/{1:}, image={2:}/{3:}, task accuracy={4:3.1f}, item accuracy={5:3.1f}'
                  .format(task, NUM_TEST_TASKS, i, num_target_images,
                          np.array(single_image_accuracies).mean() * 100.0,
                          np.array(single_task_accuracies).mean() * 100.0 if len(single_task_accuracies) > 0 else 0.0),
                  end='')
            logger.print_and_log('\r', end='')
        single_task_accuracy = np.array(single_image_accuracies).mean()
        single_task_accuracies.append(single_task_accuracy)

        _, group_task_accuracy = model.compute_objective(context_images, context_labels, target_images,
                                                           target_labels, accuracy=True)
        group_task_accuracy = group_task_accuracy.detach()
        group_task_accuracies.append(group_task_accuracy.item())

    single_accuracy = np.array(single_task_accuracies).mean() * 100.0
    single_accuracy_confidence = (196.0 * np.array(single_task_accuracies).std()) / np.sqrt(len(single_task_accuracies))
    logger.print_and_log('\nSingle: {0:3.1f}+/-{1:2.1f}'.format(single_accuracy, single_accuracy_confidence))
    group_accuracy = np.array(group_task_accuracies).mean() * 100.0
    group_accuracy_confidence = (196.0 * np.array(group_task_accuracies).std()) / np.sqrt(len(group_task_accuracies))
    logger.print_and_log('\nGroup: {0:3.1f}+/-{1:2.1f}'.format(group_accuracy, group_accuracy_confidence))


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
    logger.print_and_log('\nSingle: {0:3.1f}+/-{1:2.1f}'.format(single_accuracy, single_accuracy_confidence))
    group_accuracy = np.array(group_task_accuracies).mean() * 100.0
    group_accuracy_confidence = (196.0 * np.array(group_task_accuracies).std()) / np.sqrt(len(group_task_accuracies))
    logger.print_and_log('\nGroup: {0:3.1f}+/-{1:2.1f}'.format(group_accuracy, group_accuracy_confidence))


def print_average_accuracy( accuracies, descriptor):
    accuracy = np.array(accuracies).mean() * 100.0
    accuracy_confidence = (196.0 * np.array(accuracies).std()) / np.sqrt(len(accuracies))
    logger.print_and_log('{0:} : {1:3.1f}+/-{2:2.1f}'.format(descriptor, accuracy, accuracy_confidence))


def save_image_pair(checkpoint_dir, adv_img, clean_img, task_no, index):
    save_image(adv_img.cpu().detach().numpy(),
               os.path.join(checkpoint_dir, 'adv_task_{}_index_{}.png'.format(task_no, index)))
    save_image(clean_img.cpu().detach().numpy(), os.path.join(checkpoint_dir, 'in_task_{}_index_{}.png'.format(task_no, index)))

def attack_swap(model, dataset, model_path, tasks, config_path, checkpoint_dir):
    # load the model
    if device.type == 'cpu':
            load_dict = torch.load(model_path, map_location='cpu')
    else:
            load_dict = torch.load(model_path)
    model.load_state_dict(load_dict)

    model.set_gradient_steps(test_gradient_steps)

    context_attack = create_attack(config_path, checkpoint_dir)
    context_attack.set_attack_mode('context')
    assert context_attack.get_shot_fraction() == 1.0
    assert context_attack.get_class_fraction() == 1.0

    target_attack = create_attack(config_path, checkpoint_dir)
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

    for task in range(tasks):
        # when testing, target_shot is just shot
        task_dict = dataset.get_test_task(way=args.num_classes, shot=args.shot, target_shot=args.target_shot)
        xc, xt_all, yc, yt_all = prepare_task(task_dict)

        # Select as many target images as context images to be used on the attack
        # The rest will be used for evaluation
        assert xc.shape[0] <= xt_all.shape[0]
        split_xt, split_yt = split_target_set(xt_all, yt_all, args.shot)
        xt = split_xt[0]
        yt = split_yt[0]

        adv_context_images, adv_context_indices = context_attack.generate(xc, yc, xt, yt, model, model.compute_logits, device)

        adv_target_images, adv_target_indices = target_attack.generate(xc, yc, xt, yt, model, model.compute_logits, device)

        tmp_adv_c_indices = [xi.item() for xi in adv_context_indices]
        tmp_adv_c_indices.sort()
        assert tmp_adv_c_indices == adv_target_indices

        if args.save_samples and task < 10:
            for i in range(len(xt)):
                save_image_pair(checkpoint_dir, adv_context_images[i], xc[i], task, i)
                save_image_pair(checkpoint_dir, adv_target_images[i], xt[i], task, i)

        gen_clean_accuracies.append(model.compute_objective(xc, yc, xt, yt, accuracy=True)[1].item())
        gen_adv_context_accuracies.append(model.compute_objective(adv_context_images, yc, xt, yt, accuracy=True)[1].item())
        gen_adv_target_accuracies.append(model.compute_objective(xc, yc, adv_target_images, yt, accuracy=True)[1].item())
        # Evaluate on independent target sets
        for s in range(1, len(split_xt)):
            clean_accuracies.append(model.compute_objective(xc, yc, split_xt[s], split_yt[s], accuracy=True)[1].item())
            clean_target_as_context_accuracies.append(model.compute_objective(xt, yt, split_xt[s], split_yt[s], accuracy=True)[1].item())
            adv_context_accuracies.append(model.compute_objective(adv_context_images, yc, split_xt[s], split_yt[s], accuracy=True)[1].item())
            adv_target_accuracies.append(model.compute_objective(split_xt[s], split_yt[s], adv_target_images, yt, accuracy=True)[1].item())
            adv_target_as_context_accuracies.append(model.compute_objective(adv_target_images, yc, split_xt[s], split_yt[s], accuracy=True)[1].item())
            adv_context_as_target_accuracies.append(model.compute_objective(split_xt[s], split_yt[s], adv_context_images, yt, accuracy=True)[1].item())

    print_average_accuracy(gen_clean_accuracies, "Gen setting: Clean accuracy")
    print_average_accuracy(gen_adv_context_accuracies, "Gen setting: Context attack accuracy")
    print_average_accuracy(gen_adv_target_accuracies, "Gen setting: Target attack accuracy")

    print_average_accuracy(clean_accuracies, "Clean accuracy")
    print_average_accuracy(clean_target_as_context_accuracies, "Clean Target as Context accuracy")
    print_average_accuracy(adv_context_accuracies, "Context attack accuracy")
    print_average_accuracy(adv_target_as_context_accuracies, "Adv Target as Context accuracy")
    print_average_accuracy(adv_target_accuracies, "Target attack accuracy")
    print_average_accuracy(adv_context_as_target_accuracies, "Adv Context as Target")


def attack(model, dataset, model_path, tasks, config_path, checkpoint_dir):
    # load the model
    if device.type == 'cpu':
            load_dict = torch.load(model_path, map_location='cpu')
    else:
            load_dict = torch.load(model_path)
    model.load_state_dict(load_dict)

    model.set_gradient_steps(test_gradient_steps)

    attack = create_attack(config_path, checkpoint_dir)

    accuracies_before = []
    accuracies_after = []
    for task in range(tasks):
        # when testing, target_shot is just shot
        task_dict = dataset.get_test_task(way=args.num_classes, shot=args.shot, target_shot=args.shot)
        xc, xt, yc, yt = prepare_task(task_dict)

        if attack.get_attack_mode() == 'context':
            adv_context_images, adv_context_indices = attack.generate(xc, yc, xt, yt, model, model.compute_logits, device)

            if args.save_samples and task < 10:
                for i in range(len(xc)):
                    save_image_pair(checkpoint_dir, adv_context_images[i], xc[i], task, i)

            _, acc_after = model.compute_objective(adv_context_images, yc, xt, yt, accuracy=True)

        else:  # target
            adv_target_images, _ = attack.generate(xc, yc, xt, yt, model, model.compute_logits, device)
            if args.save_samples and task < 10:
                for i in range(len(xt)):
                    save_image_pair(checkpoint_dir, adv_target_images[i], xt[i], task, i)

            _, acc_after = model.compute_objective(xc, yc, adv_target_images, yt, accuracy=True)

        _, acc_before = model.compute_objective(xc, yc, xt, yt, accuracy=True)

        accuracies_before.append(acc_before.item())
        accuracies_after.append(acc_after.item())

    print_average_accuracy(accuracies_before, "Before attack")
    print_average_accuracy(accuracies_after, "After attack")


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
parser.add_argument("--swap_attack", default=False,
                    help="When attacking, should the attack be a swap attack or not.")
parser.add_argument("--save_samples", default=False,
                    help="Output samples of the clean and adversarial images")
args = parser.parse_args()

# Create checkpoint directory (if required)
if not os.path.exists(args.checkpoint_dir):
    os.makedirs(args.checkpoint_dir)

logger = Logger(args.checkpoint_dir, 'log.txt')

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
            logger.print_and_log("Loss is inf, avoiding optimization step")
        else:
            optimizer.step()

        if (iteration + 1) % PRINT_FREQUENCY == 0:
            # print training stats
            logger.print_and_log('Task [{}/{}], Train Loss: {:.7f}, '\
                  'Train Acc: {:.7f}'.format(iteration + 1,
                                             args.iterations,
                                             loss.item(),
                                             accuracy.item()))

        if (iteration + 1) % VALIDATE_FREQUENCY == 0:
            val_loss, val_accuracy, confidence = validate(model, data,
                                                          batch_size=128)

            logger.print_and_log('Validating at [{}/{}], Validation Loss: {:.7f}, '\
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
                logger.print_and_log("Best validation model updated.")

    # save the fully trained model
    torch.save(model.state_dict(), os.path.join(args.checkpoint_dir,
                                                'fully_trained.pt'))

    # After training -- test the model on test set
    test(model, data, os.path.join(args.checkpoint_dir, 'fully_trained.pt'))
    test(model, data, os.path.join(args.checkpoint_dir, 'best_validation.pt'))

elif args.mode == 'attack':
    if not args.swap_attack:
        attack(model, data, args.attack_model_path, args.attack_tasks, args.attack_config_path, args.checkpoint_dir)
    else:
        attack_swap(model, data, args.attack_model_path, args.attack_tasks, args.attack_config_path, args.checkpoint_dir)

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
