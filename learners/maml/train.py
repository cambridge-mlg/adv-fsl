import argparse
import torch
import os
import numpy as np

from pytorch.src.mini_imagenet import MiniImageNetData, prepare_task
from pytorch.src.omniglot import OmniglotData
from pytorch.src.maml import MAML
from pytorch.src.proto_maml import ProtoMAMLv2 as ProtoMAML
from pytorch.src.shrinkage_maml import SigmaMAML as sMAML
from pytorch.src.shrinkage_maml import PredCPMAML as pMAML

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
                    choices=['train_and_test', 'test_only'],
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
    test_gradient_steps = 3

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
                 flatten=flatten)

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


else:  # test only
    # Test fully trained
    args.model_path = os.path.join(args.checkpoint_dir, 'fully_trained.pt')
    test(model, data, args.model_path)
    # Test best validation model
    args.model_path = os.path.join(args.checkpoint_dir, 'best_validation.pt')
    test(model, data, args.model_path)
