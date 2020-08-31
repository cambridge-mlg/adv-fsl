import argparse
import matplotlib.pyplot as plt
import numpy as np
import torch
import os
from PIL import Image
from libs.Universal_Adversarial_Perturbation.feature_extractor import create_feature_extractor
from libs.Universal_Adversarial_Perturbation.classifier import Classifier
from libs.Universal_Adversarial_Perturbation.transforms import transform
from libs.Universal_Adversarial_Perturbation.generate import generate

"""
Command line parser
"""


def parse_command_line():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_path", default="../datasets", help="Path to dataset records.")
    parser.add_argument("--feature_extractor", choices=["mnasnet", "resnet", "maml_convnet", "protonets_convnet"],
                        default="resnet", help="Dataset to use.")
    parser.add_argument("--pretrained_feature_extractor_path",
                        default="E:/repos/adv-fsl/learners/cnaps/models/pretrained_resnet.pt.tar",
                        help="Path to pretrained feature extractor model.")
    parser.add_argument("--epsilon", type=int, default=10, help="Largest +/- gray level shift on a 0-255 scale.")

    args = parser.parse_args()

    return args


def main():
    args = parse_command_line()

    # get device
    gpu_device = 'cuda:0'
    device = torch.device(gpu_device if torch.cuda.is_available() else 'cpu')

    # create the network
    num_classes = 712  # train partition of meta-dataset
    if args.feature_extractor == "maml_convnet" or args.feature_extractor == "protonets_convnet":  # mini-imagenet
        num_classes = 64

    feature_extractor = create_feature_extractor(args.feature_extractor, args.pretrained_feature_extractor_path)
    net = Classifier(num_classes, feature_extractor)
    net.to(device)
    net.eval()

    # compute perturbation
    v = generate(args.data_path, 'train.txt', 'test.txt', device, net, max_iter_uni=1000, delta=0.2, p=np.inf,
                 num_classes=10, overshoot=0.2, max_iter_df=10, xi=args.epsilon)
    # Saving the universal perturbation
    np.save(os.path.join(args.data_path, 'universal.npy'), v)


# testimg = "./data/test_im2.jpg"
# print('>> Testing the universal perturbation on',testimg)
# labels = open('./data/labels.txt', 'r').read().split('\n')
# testimgToInput = Image.open(testimg).convert('RGB')
# pertimgToInput = np.clip(cut(testimgToInput)+v,0,255)
# pertimg = Image.fromarray(pertimgToInput.astype(np.uint8))
#
# img_orig = transform(testimgToInput)
# inputs_orig = img_orig[np.newaxis, :].to(device)
# outputs_orig = net(inputs_orig)
# _, predicted_orig = outputs_orig.max(1)
# label_orig = labels[predicted_orig[0]]
#
# img_pert=transform(pertimg)
# inputs_pert=img_pert[np.newaxis, :].to(device)
# outputs_pert=net(inputs_pert)
# _, predicted_pert = outputs_pert.max(1)
# label_pert=labels[predicted_pert[0]]
#
# # Show original and perturbed image
# plt.figure()
# plt.subplot(1, 2, 1)
# plt.imshow(cut(testimgToInput), interpolation=None)
# plt.title(label_orig)
#
# plt.subplot(1, 2, 2)
# plt.imshow(pertimg, interpolation=None)
# plt.title(label_pert)
#
# plt.savefig("./data/result.png")
# plt.show()


if __name__ == "__main__":
    main()
