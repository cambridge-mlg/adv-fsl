import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
import matplotlib.pyplot as plt
import numpy as np


def plot_curves(file_list, labels, line_styles, output_file):
    fig = plt.figure()
    for file, label, line_style in zip(file_list, labels, line_styles):
        data = np.genfromtxt(file, delimiter=',', names=['Size', "Accuracy"])
        plt.plot(data['Size'], data['Accuracy'], label=label, linestyle=line_style)

    plt.xlabel('Seed Query Set Size', fontsize='x-large')
    plt.xlim(0, 500)
    plt.ylabel('% Drop in Accuracy', fontsize='x-large')
    plt.legend(loc='best', fontsize='x-large')
    plt.savefig(output_file, bbox_inches='tight')
    plt.close()


def main():
    line_styles = [
        ':', '-', '--', '-.'
    ]

    file_list = [
        './data/cnaps_all_1-shot.txt',
        './data/cnaps_single_1-shot.txt',
        './data/protonets_all_1-shot.txt',
        './data/protonets_single_1-shot.txt'
    ]

    labels = [
        'CNAPs, All, 1-shot',
        'CNAPs, Single, 1-shot',
        'ProtoNets, All, 1-shot',
        'ProtoNets, Single, 1-shot'
    ]

    plot_curves(file_list=file_list, labels=labels, line_styles=line_styles,
                output_file='./plots/accuracy_vs_query_examples_1_shot.pdf')

    file_list = [
        './data/cnaps_all_5-shot.txt',
        './data/cnaps_single_5-shot.txt',
        './data/protonets_all_5-shot.txt',
        './data/protonets_single_5-shot.txt'
    ]

    labels = [
        'CNAPs, All, 5-shot',
        'CNAPs, Single, 5-shot',
        'ProtoNets, All, 5-shot',
        'ProtoNets, Single, 5-shot'
    ]

    plot_curves(file_list=file_list, labels=labels, line_styles=line_styles,
                output_file='./plots/accuracy_vs_query_examples_5_shot.pdf')


if __name__ == '__main__':
    main()
