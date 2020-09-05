import argparse
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


"""
Command line parser
"""


def parse_command_line():
    parser = argparse.ArgumentParser()

    parser.add_argument("--in_path", help="Path to uap image in .npy format.")
    parser.add_argument("--out_path", help="Path to output visualization in PDF format.")

    args = parser.parse_args()

    return args


def main():
    args = parse_command_line()

    # load uap image
    uap_image = np.load(args.in_path)
    min = np.min(uap_image)
    max = np.max(uap_image)
    shifted_image = uap_image - min
    scaled_image = 255.0 * shifted_image / (max - min)

    shifted_im = Image.fromarray(shifted_image.astype(np.uint8))
    scaled_im = Image.fromarray(scaled_image.astype(np.uint8))

    # Show original and perturbed image
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(shifted_im, interpolation=None)

    plt.subplot(1, 2, 2)
    plt.imshow(scaled_im, interpolation=None)

    plt.savefig(args.out_path)
    plt.show()


if __name__ == "__main__":
    main()
