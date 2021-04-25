import torch
import torch.nn as nn
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
import argparse
import os
from extras.resnet import resnet18, resnet34
from extras.vgg import vgg11_bn


def save_checkpoint(state, save_path):
    filename = os.path.join(save_path, 'checkpoint.pth.tar')
    torch.save(state, filename)


def save(model, model_path):
    torch.save(model.state_dict(), model_path)


def accuracy(logits, labels):
    predictions = torch.argmax(logits, dim=1)
    return torch.mean(torch.eq(predictions, labels).float())


class AverageMeter(object):
    def __init__(self):
        self.avg = 0
        self.sum = 0
        self.cnt = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt


def adjust_learning_rate(optimizer, epoch, initial_learning_rate, decay_period):
    """Sets the learning rate to the initial LR decayed by 10 every args.decay epochs"""
    lr = initial_learning_rate * (0.1 ** (epoch // decay_period))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


class Net(nn.Module):
    def __init__(self, net, num_classes):
        super().__init__()
        if net == "resnet18":
            self.feature_extractor = resnet18()
        elif net == "resnet34":
            self.feature_extractor = resnet34()
        else:
            self.feature_extractor = vgg11_bn()

        num_features = self.feature_extractor.output_size
        self.fc = nn.Linear(num_features, num_classes)

    def encode(self, x):
        return self.feature_extractor(x)

    def forward(self, x):
        h = self.encode(x)
        return self.fc(h)


def train(train_queue, model, criterion, optimizer, grad_clip, report_freq):
    objs = AverageMeter()
    top1 = AverageMeter()
    model.train()

    total_step = len(train_queue)
    for step, (input, target) in enumerate(train_queue):
        target = target.cuda()
        input = input.cuda()

        optimizer.zero_grad()
        logits = model(input)
        loss = criterion(logits, target)

        loss.backward()
        nn.utils.clip_grad_norm(model.parameters(), grad_clip)
        optimizer.step()

        prec1 = accuracy(logits, target)
        n = input.size(0)
        objs.update(loss.item(), n)
        top1.update(prec1, n)

        if (step + 1) % report_freq == 0:
            print('train: step %03d of %03d loss: %f acc: %f' % (step, total_step, objs.avg, top1.avg))

    return top1.avg, objs.avg


def infer(valid_queue, model, criterion, report_freq):
    objs = AverageMeter()
    top1 = AverageMeter()
    model.eval()

    with torch.no_grad():
        total_step = len(valid_queue)
        for step, (input, target) in enumerate(valid_queue):
            target = target.cuda()
            input = input.cuda()

            logits = model(input)
            loss = criterion(logits, target)

            prec1 = accuracy(logits, target)
            n = input.size(0)
            objs.update(loss.data.item(), n)
            top1.update(prec1, n)

            if (step + 1) % report_freq == 0:
                print('valid step %03d of %03d loss: %f acc; %f' % (step, total_step, objs.avg, top1.avg))

    return top1.avg, objs.avg


def parse_command_line():
    parser = argparse.ArgumentParser()
    # dataset options
    parser.add_argument("--data_root", type=str, default='/scratch/jg801/tiered-imagenet/flat/',
                        help="Root directory of training data")
    parser.add_argument("--training_proportion", type=float, default=0.8,
                        help="Proportion of (training) data to use for pretraining")
    parser.add_argument("--net", choices=["vgg11_bn", "resnet18", "resnet34"], default=vgg11_bn, help="Net to train")
    # Optimization hyper-parameters
    parser.add_argument('--batch_size', type=int, default=256, help='batch size')
    parser.add_argument('--epochs', type=int, default=250, help='num of training epochs')
    parser.add_argument('--learning_rate', type=float, default=0.1, help='init learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay')
    parser.add_argument('--report_freq', type=float, default=100, help='report frequency')
    parser.add_argument('--decay_period', type=int, default=50, help='epochs between two learning rate decays')
    parser.add_argument('--grad_clip', type=float, default=5., help='gradient clipping')
    # Saving and loading
    parser.add_argument('--save_path', type=str, default='./models/', help='experiment name')

    args = parser.parse_args()
    return args


def main():
    args = parse_command_line()

    # Define dataset and
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    data = ImageFolder(
        root=args.data_root,
        transform=transforms.Compose([
            transforms.RandomCrop(84, padding=8),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(
                brightness=0.4,
                contrast=0.4,
                saturation=0.4,
                hue=0.2),
            transforms.ToTensor(),
            normalize,
        ]))

    # Split the data
    N = len(data)
    train_proportion, test_proportion = args.training_proportion, 1.0 - args.training_proportion
    ntrain = int(train_proportion * N)
    splits = (ntrain, N - ntrain)
    train_data, test_data = torch.utils.data.random_split(data, splits)
    train_queue = torch.utils.data.DataLoader(
        dataset=train_data,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4
    )
    test_queue = torch.utils.data.DataLoader(
        dataset=test_data,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4
    )

    # Define model, optimizer, etc'
    num_classes = 712  # number of meta-dataset imagenet training classes
    model = Net(args.net, num_classes)
    model.cuda()

    loss = nn.CrossEntropyLoss()
    loss.cuda()

    optimizer = torch.optim.SGD(
        model.parameters(),
        args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay
    )

    # Train model
    best_validation_accuracy = 0.0
    for epoch in range(args.epochs):
        adjust_learning_rate(optimizer, epoch, args.learning_rate, args.decay_period)

        train_acc, train_obj = train(train_queue, model, loss, optimizer, args.grad_clip, args.report_freq)
        print('Epoch [{}/{}]: train loss: {} train_acc: {} lr:{}'.format(epoch + 1, args.epochs, train_obj, train_acc, optimizer.param_groups[0]['lr']))

        val_acc, val_obj = infer(test_queue, model, loss, args.report_freq)
        print('Epoch [{}/{}]: validation loss {}, validation accuracy {}'.format(epoch + 1, args.epochs, val_obj, val_acc))

        if val_acc > best_validation_accuracy:
            best_validation_accuracy = val_acc
            torch.save(model.feature_extractor.state_dict(),
                       os.path.join(args.save_path, 'pretrained_{}_best_validation.pt'.format(args.net)))
            print('Best validation model updated.')

        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.feature_extractor.state_dict(),
            'best_acc_top1': best_validation_accuracy,
            'optimizer': optimizer.state_dict(),
        }, args.save_path)

    torch.save(model.feature_extractor.state_dict(),
               os.path.join(args.save_path, 'pretrained_{}_fully_trained.pt'.format(args.net)))


if __name__ == '__main__':
    main()
