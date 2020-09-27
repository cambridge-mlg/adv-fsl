import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
import matplotlib.pyplot as plt
import numpy as np


def plot_curves(file_list, labels, line_styles, colors, output_file):
    fig = plt.figure()
    for file, label, line_style, color in zip(file_list, labels, line_styles, colors):
        data = np.genfromtxt(file, delimiter=',', names=['Size', "Accuracy"])
        plt.plot(data['Size'], data['Accuracy'], label=label, linestyle=line_style, color=color)

    plt.xlabel('Seed Query Set Size', fontsize='x-large')
    plt.xlim(0, 500)
    plt.ylim(50, 80)
    plt.ylabel('Decrease in Accuracy (%)', fontsize='x-large')
    plt.legend(loc='lower right', fontsize='large')
    plt.savefig(output_file, bbox_inches='tight')
    plt.close()


def main():
    colors = [
        '#DE3163', '#DE3163', '#64C3EB', '#64C3EB', 'purple', 'purple'
    ]
    line_styles = [
        '-', '--', '-', '--', '-', '--'
    ]

    file_list = [
        # './data/cnaps_max_1-shot.txt',
        # './data/cnaps_max_5-shot.txt',
        './data/protonets_max_1-shot.txt',
        './data/protonets_max_5-shot.txt',
        './data/maml_max_1-shot.txt',
        './data/maml_max_5-shot.txt'
    ]

    labels = [
        # 'CNAPs 1-shot',
        # 'CNAPs 5-shot',
        'ProtoNets 1-shot',
        'ProtoNets 5-shot',
        'MAML 1-shot',
        'MAML 5-shot'
    ]

    plot_curves(file_list=file_list, labels=labels, line_styles=line_styles, colors=colors,
                output_file='./plots/accuracy_vs_query_examples.pdf')

    # file_list = [
    #     './data/cnaps_max_5-shot.txt',
    #     './data/protonets_max_5-shot.txt',
    #     './data/maml_max_5-shot.txt'
    # ]
    #
    # labels = [
    #     'CNAPs',
    #     'ProtoNets',
    #     'MAML',
    # ]
    #
    # plot_curves(file_list=file_list, labels=labels, line_styles=line_styles, colors=colors,
    #             output_file='./plots/accuracy_vs_query_examples_5_shot.pdf')


if __name__ == '__main__':
    main()
