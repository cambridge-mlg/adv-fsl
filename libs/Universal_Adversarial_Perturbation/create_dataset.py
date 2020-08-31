import argparse
import os
import numpy as np
from shutil import copyfile


parser = argparse.ArgumentParser()
parser.add_argument("--source_dir", help="Path to source images")
parser.add_argument("--target_dir", help="Path to destination images")
parser.add_argument("--train_images_per_class", type=int, default=14,
                    help="Number of train images per class to sample from source_dir and copy to target_dir.")
parser.add_argument("--test_images_per_class", type=int, default=2,
                    help="Number of test images per class to sample from source_dir and copy to target_dir.")
args = parser.parse_args()

if __name__ == '__main__':
    # create the target directories
    if not os.path.exists(args.target_dir):
        os.makedirs(args.target_dir)
    train_dir = os.path.join(args.target_dir, 'train')
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)
    test_dir = os.path.join(args.target_dir, 'test')
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)

    # open train and test file lists for writing
    train_file = open(os.path.join(args.target_dir, "train.txt"), "w", buffering=1)
    test_file = open(os.path.join(args.target_dir, "test.txt"), "w", buffering=1)

    for i, dir in enumerate(os.listdir(args.source_dir)):
        current_source_dir = os.path.join(args.source_dir, dir)
        file_names = os.listdir(current_source_dir)
        num_files = len(file_names)
        num_images_to_sample = args.train_images_per_class + args.test_images_per_class
        sample_indices = np.random.choice(num_files, size=num_images_to_sample, replace=False)

        # copy the train images
        for index in sample_indices[0:args.train_images_per_class]:
            image_name = file_names[index]
            image_path = os.path.join(train_dir, image_name)
            copyfile(os.path.join(current_source_dir, image_name), image_path)
            train_file.write("{} {}\n".format(image_name, i))

        # copy the test images
        for index in sample_indices[args.train_images_per_class:]:
            image_name = file_names[index]
            image_path = os.path.join(test_dir, image_name)
            copyfile(os.path.join(current_source_dir, image_name), image_path)
            test_file.write("{} {}\n".format(image_name, i))

    # close the files
    train_file.close()
    test_file.close()
