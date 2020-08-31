import argparse
import os
from PIL import Image
import numpy as np
import pickle


def process_data(in_dir, out_dir, split, text_file_name, binary_file_name, num_images):
    input_text_file = open(os.path.join(in_dir, text_file_name), "r", buffering=1)
    output_text_file = open(os.path.join(out_dir, text_file_name), "w", buffering=1)
    output = np.zeros((num_images, 84, 84, 3), dtype=np.uint8)

    for i, line in enumerate(input_text_file):
        tokens = line.split()
        label = tokens[1]
        output_text_file.write("{} {}\n".format(i, label))
        image = Image.open(os.path.join(in_dir, split, tokens[0])).convert('RGB')
        output[i] = image

    input_text_file.close()
    output_text_file.close()

    # persist the opened images
    pickle_file = open(os.path.join(out_dir, binary_file_name), 'wb')
    pickle.dump(output, pickle_file, protocol=pickle.HIGHEST_PROTOCOL)
    pickle_file.close()


def parse_command_line():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_dir", help="Path to source images")
    parser.add_argument("--target_dir", help="Path to destination images")
    parser.add_argument("--train_images_per_class", type=int, default=14,
                        help="Number of train images per class to sample from source_dir and copy to target_dir.")
    parser.add_argument("--test_images_per_class", type=int, default=2,
                        help="Number of test images per class to sample from source_dir and copy to target_dir.")
    parser.add_argument("--num_classes", type=int, default=712,
                        help="Number of classes in the data.")
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_command_line()

    # create the target directories
    if not os.path.exists(args.target_dir):
        os.makedirs(args.target_dir)

    # process data
    process_data(args.source_dir, args.target_dir, 'train', 'train.txt', 'train.pkl',
                 args.train_images_per_class * args.num_classes)  # train data
    process_data(args.source_dir, args.target_dir, 'test', 'test.txt', 'test.pkl',
                 args.test_images_per_class * args.num_classes)  # test data


