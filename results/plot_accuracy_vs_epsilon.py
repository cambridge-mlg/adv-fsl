import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
import matplotlib.pyplot as plt
import numpy as np


def plot_curves(file_list, labels, colors, line_styles, markers, output_file, legend_location):
    fig = plt.figure()
    for file, label, color, line_style, marker in zip(file_list, labels, colors, line_styles, markers):
        data = np.genfromtxt(file, delimiter=',', names=['Epsilon', 'Accuracy', "Error"])
        plt.plot(data['Epsilon'], data['Accuracy'], color=color, label=label, linestyle=line_style, marker=marker)
        plt.fill_between(data['Epsilon'], data['Accuracy'] - data['Error'],
                         data['Accuracy'] + data['Error'], color=color, alpha=0.2)

    plt.xlabel('Epsilon', fontsize='x-large')
    plt.ylabel('Accuracy (%)', fontsize='x-large')
    plt.legend(loc=legend_location, fontsize='large')
    plt.savefig(output_file, bbox_inches='tight')
    plt.close()


def main():
    file_list = [
        './results/data/adv_train_natural_context_protonets_5_1.txt',
        './results/data/adv_train_natural_context_protonets_5_5.txt',
        './results/data/adv_train_support_context_protonets_5_1.txt',
        './results/data/adv_train_support_context_protonets_5_5.txt'
    ]

    labels = [

        'Natural - 1 shot',
        'Natural - 5 shot',
        'Adv Support - 1 shot',
        'Adv Support - 5 shot'
    ]

    colors = [
        'red',
        'blue',
        'red',
        'blue'
    ]

    line_styles = [
        '-', '-', '--', '--'
    ]

    markers = [
        '.', '.', 's', 's'
    ]

    plot_curves(file_list=file_list, labels=labels, colors=colors, line_styles=line_styles, markers=markers,
                output_file='./results/plots/accuracy_vs_epsilon_adv_support.pdf', legend_location='upper right')

    file_list = [
        './results/data/adv_train_natural_target_protonets_5_1.txt',
        './results/data/adv_train_natural_target_protonets_5_5.txt',
        './results/data/adv_train_support_target_protonets_5_1.txt',
        './results/data/adv_train_support_target_protonets_5_5.txt'
    ]

    plot_curves(file_list=file_list, labels=labels, colors=colors, line_styles=line_styles, markers=markers,
                output_file='./results/plots/accuracy_vs_epsilon_adv_query.pdf', legend_location='upper right')


if __name__ == '__main__':
    main()
