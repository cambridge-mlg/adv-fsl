import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
import matplotlib.pyplot as plt
import numpy as np


def plot_curves(file_list, labels, colors, line_styles, markers, output_file, legend_location):
    fig = plt.figure()
    for file, label, color, line_style, marker in zip(file_list, labels, colors, line_styles, markers):
        data = np.genfromtxt(file, delimiter=',', names=['Epsilon', 'Accuracy'])
        plt.plot(data['Epsilon'], data['Accuracy'], color=color, label=label, linestyle=line_style, marker=marker)

    plt.xlabel('Fraction of Adversarial Instances', fontsize='x-large')
    plt.ylabel('Accuracy (%)', fontsize='x-large')
    plt.legend(loc=legend_location, fontsize='medium', bbox_to_anchor=(1.05, 1), )
    plt.savefig(output_file, bbox_inches='tight')
    plt.close()


def main():
    file_list = [
        './results/data/meta-dataset_protonets_ilsvrc_general.txt',
        './results/data/meta-dataset_protonets_aircraft_general.txt',
        './results/data/meta-dataset_protonets_birds_general.txt',
        './results/data/meta-dataset_protonets_quick_draw_general.txt',
        './results/data/meta-dataset_protonets_vgg_flower_general.txt',
        './results/data/meta-dataset_protonets_mscoco_general.txt',
        './results/data/meta-dataset_protonets_mnist_general.txt',
        './results/data/meta-dataset_protonets_cfar10_general.txt',
        './results/data/meta-dataset_protonets_cfar100_general.txt',
        './results/data/meta-dataset_protonets_ilsvrc_swap.txt',
        './results/data/meta-dataset_protonets_aircraft_swap.txt',
        './results/data/meta-dataset_protonets_birds_swap.txt',
        './results/data/meta-dataset_protonets_quick_draw_swap.txt',
        './results/data/meta-dataset_protonets_vgg_flower_swap.txt',
        './results/data/meta-dataset_protonets_mscoco_swap.txt',
        './results/data/meta-dataset_protonets_mnist_swap.txt',
        './results/data/meta-dataset_protonets_cfar10_swap.txt',
        './results/data/meta-dataset_protonets_cfar100_swap.txt',

    ]

    labels = [
        'ILSVRC - General',
        'Aircraft - General',
        'Birds - General',
        'Quick Draw - General',
        'VGG Flower General',
        'MSCOCO General',
        'MNIST General',
        'CIFAR10 General',
        'CIFAR100 General',
        'ILSVRC - Swap',
        'Aircraft - Swap',
        'Birds - Swap',
        'Quick Draw - Swap',
        'VGG Flower Swap',
        'MSCOCO Swap',
        'MNIST Swap',
        'CIFAR10 Swap',
        'CIFAR100 Swap'
    ]

    colors = [
        'red',
        'blue',
        'green',
        'black',
        'yellow',
        'purple',
        'orange',
        'brown',
        'cyan',
        'red',
        'blue',
        'green',
        'black',
        'yellow',
        'purple',
        'orange',
        'brown',
        'cyan',
    ]

    line_styles = [
        '-', '-', '-', '-', '-', '-', '-', '-', '-',
        '--', '--', '--', '--', '--', '--', '--', '--', '--',
    ]

    markers = [
        '.', '.', '.', '.', '.', '.', '.', '.', '.',
        '.', '.', '.', '.', '.', '.', '.', '.', '.',
    ]

    plot_curves(file_list=file_list, labels=labels, colors=colors, line_styles=line_styles, markers=markers,
                output_file='./results/plots/meta-dataset.pdf', legend_location='upper left')


if __name__ == '__main__':
    main()
