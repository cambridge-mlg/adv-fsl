import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.utils.data as data
import pickle
import argparse
import os
from learners.fine_tune.src.convnet import MamlNet, ProtoNets


class MiniImageNetDataset(data.Dataset):
    def __init__(self, path):
        self.train_set = pickle.load(open(path, 'rb'))
        # normalize to -1 to 1
        normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                         std=[0.5, 0.5, 0.5])

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            normalize
        ])

        self.num_classes = 64
        self.num_images_per_class = 600
        self.num_images = self.num_classes * self.num_images_per_class

    def __getitem__(self, index):
        class_index = index // self.num_images_per_class
        image_index = index % self.num_images_per_class

        image = self.transform(self.train_set[class_index][image_index])

        return image, class_index

    def __len__(self):
        return self.num_images


def adjust_learning_rate(optimizer, epoch, initial_learning_rate):
    """Sets the learning rate to the initial LR decayed by 10 every args.decay epochs"""
    lr = initial_learning_rate * (0.5 ** (epoch // 10))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def parse_command_line():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", choices=['maml', "protonets"], default="maml", help="Model to train.")
    parser.add_argument("--data_path", help="Path to mini-imagenet training data.")
    parser.add_argument("--save_path", help="Path to save pretrained model.")
    args = parser.parse_args()
    return args


def main():
    args = parse_command_line()

    # Device configuration
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Hyper parameters
    num_epochs = 100
    batch_size = 256
    learning_rate = 0.1

    dataset = MiniImageNetDataset(os.path.join(args.data_path, 'mini_imagenet_train.pkl'))

    # Split the data
    N = len(dataset)
    train_proportion = 0.9
    num_train = int(train_proportion * N)
    splits = (num_train, N - num_train)
    train_data, test_data = data.random_split(dataset, splits)
    train_loader = data.DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True, num_workers=1)
    test_loader = data.DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False, num_workers=1)

    if args.model == "protonets":
        model = ProtoNets().to(device)
    else:
        model = MamlNet().to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    # optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    optimizer = torch.optim.SGD(
        model.parameters(),
        learning_rate,
        momentum=0.9,
        weight_decay=1e-3
    )

    # Train the model
    best_validation_accuracy = 0.0
    total_step = len(train_loader)
    for epoch in range(num_epochs):
        adjust_learning_rate(optimizer, epoch, learning_rate)
        for i, (images, labels) in enumerate(train_loader):
            model.train()
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i + 1) % 50 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, lr: {:.4f}'
                      .format(epoch + 1, num_epochs, i + 1, total_step, loss.item(), optimizer.param_groups[0]['lr']))

        model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            for images, labels in test_loader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            validation_accuracy = 100 * correct / total
            print('Validation Accuracy: {:.2f}'.format(validation_accuracy))
            if validation_accuracy > best_validation_accuracy:
                best_validation_accuracy = validation_accuracy
                torch.save(model.feature_extractor.state_dict(),
                           os.path.join(args.save_path, 'pretrained_conv_net_best_validation.pt'))
                print('Best validation model updated.')

    # Save the model checkpoint
    torch.save(model.feature_extractor.state_dict(),
               os.path.join(args.save_path, 'pretrained_conv_net_fully_trained.pt'))


if __name__ == "__main__":
    main()
