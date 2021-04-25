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
    plt.legend(loc=legend_location, fontsize='medium')
    plt.savefig(output_file, bbox_inches='tight')
    plt.close()


def main():
    labels = [
        'AS',
        'AQ',
        'Natural'
    ]

    colors = [
        'blue',
        'red',
        'black'
    ]

    line_styles = [
        '--', ':', '-'
    ]

    markers = [
        's', '^', '.'
    ]

    file_list_protonets_AS_1 = [
        './results/data/adv_train_support_context_protonets_5_1.txt',
        './results/data/adv_train_query_context_protonets_5_1.txt',
        './results/data/adv_train_natural_context_protonets_5_1.txt'
    ]

    file_list_protonets_AQ_1 = [
        './results/data/adv_train_support_target_protonets_5_1.txt',
        './results/data/adv_train_query_target_protonets_5_1.txt',
        './results/data/adv_train_natural_target_protonets_5_1.txt'
    ]

    file_list_protonets_AS_5 = [
        './results/data/adv_train_support_context_protonets_5_5.txt',
        './results/data/adv_train_query_context_protonets_5_5.txt',
        './results/data/adv_train_natural_context_protonets_5_5.txt'
    ]

    file_list_protonets_AQ_5 = [
        './results/data/adv_train_support_target_protonets_5_5.txt',
        './results/data/adv_train_query_target_protonets_5_5.txt',
        './results/data/adv_train_natural_target_protonets_5_5.txt'
    ]

    file_lists = [
        file_list_protonets_AS_1,
        file_list_protonets_AQ_1,
        file_list_protonets_AS_5,
        file_list_protonets_AQ_5
    ]

    titles = [
        'ASP Attack - 1-shot',
        'Query Attack - 1-shot',
        'ASP Attack - 5-shot',
        'Query Attack - 5-shot'
    ]

    fig, axs = plt.subplots(2, 2, sharex=True, sharey=True)

    for file_list, title, ax in zip(file_lists, titles, [axs[0, 0], axs[0, 1], axs[1, 0], axs[1, 1]]):
        for file, label, color, line_style, marker in zip(file_list, labels, colors, line_styles, markers):
            data = np.genfromtxt(file, delimiter=',', names=['Epsilon', 'Accuracy', "Error"])
            ax.plot(data['Epsilon'], data['Accuracy'], color=color, label=label, linestyle=line_style, marker=marker)
            ax.set_title(title)
            ax.legend(loc='upper right', fontsize='small')

        axs[1,0].set_xlabel('Epsilon', fontsize='x-large')
        axs[1,1].set_xlabel('Epsilon', fontsize='x-large')
        axs[0, 0].set_ylabel('Accuracy (%)', fontsize='x-large')
        axs[1, 0].set_ylabel('Accuracy (%)', fontsize='x-large')
    plt.savefig('./results/plots/accuracy_vs_epsilon_adv.pdf', bbox_inches='tight')
    plt.close()

    file_list_maml_AS_1 = [
        './results/data/adv_train_support_context_maml_5_1.txt',
        './results/data/adv_train_query_context_maml_5_1.txt',
        './results/data/adv_train_natural_context_maml_5_1.txt'
    ]

    file_list_maml_AQ_1 = [
        './results/data/adv_train_support_target_maml_5_1.txt',
        './results/data/adv_train_query_target_maml_5_1.txt',
        './results/data/adv_train_natural_target_maml_5_1.txt'
    ]

    file_list_maml_AS_5 = [
        './results/data/adv_train_support_context_maml_5_5.txt',
        './results/data/adv_train_query_context_maml_5_5.txt',
        './results/data/adv_train_natural_context_maml_5_5.txt'
    ]

    file_list_maml_AQ_5 = [
        './results/data/adv_train_support_target_maml_5_5.txt',
        './results/data/adv_train_query_target_maml_5_5.txt',
        './results/data/adv_train_natural_target_maml_5_5.txt'
    ]

    file_lists = [
        file_list_maml_AS_1,
        file_list_maml_AQ_1,
        file_list_maml_AS_5,
        file_list_maml_AQ_5
    ]

    fig, axs = plt.subplots(2, 2, sharex=True, sharey=True)

    for file_list, title, ax in zip(file_lists, titles, [axs[0, 0], axs[0, 1], axs[1, 0], axs[1, 1]]):
        for file, label, color, line_style, marker in zip(file_list, labels, colors, line_styles, markers):
            data = np.genfromtxt(file, delimiter=',', names=['Epsilon', 'Accuracy', "Error"])
            ax.plot(data['Epsilon'], data['Accuracy'], color=color, label=label, linestyle=line_style, marker=marker)
            ax.set_title(title)
            if ax == axs[0, 1]:
                ax.legend(loc='upper right', fontsize=10)

        axs[1, 0].set_xlabel('Epsilon', fontsize='x-large')
        axs[1, 1].set_xlabel('Epsilon', fontsize='x-large')
        axs[0, 0].set_ylabel('Accuracy (%)', fontsize='x-large')
        axs[1, 0].set_ylabel('Accuracy (%)', fontsize='x-large')

    # axs[0, 0].legend(ncol=len(labels), bbox_to_anchor=(0, 1.1), loc='lower left', fontsize='large')
    plt.savefig('./results/plots/accuracy_vs_epsilon_adv_maml.pdf', bbox_inches='tight')
    plt.close()


if __name__ == '__main__':
    main()