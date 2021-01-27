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

        'Natural',
        'AS',
        'AQ'
    ]

    colors = [
        'black',
        'blue',
        'red'
    ]

    line_styles = [
        '-', '--', ':'
    ]

    markers = [
        '.', 's', '^'
    ]

    file_list_AS_1 = [
        './results/data/adv_train_natural_context_protonets_5_1.txt',
        './results/data/adv_train_support_context_protonets_5_1.txt',
        './results/data/adv_train_query_context_protonets_5_1.txt'
    ]

    # plot_curves(file_list=file_list_AS_1, labels=labels, colors=colors, line_styles=line_styles, markers=markers,
    #             output_file='./results/plots/accuracy_vs_epsilon_adv_support-1-shot.pdf', legend_location='upper right')

    file_list_AQ_1 = [
        './results/data/adv_train_natural_target_protonets_5_1.txt',
        './results/data/adv_train_support_target_protonets_5_1.txt',
        './results/data/adv_train_query_target_protonets_5_1.txt',
    ]

    # plot_curves(file_list=file_list_AQ_1, labels=labels, colors=colors, line_styles=line_styles, markers=markers,
    #             output_file='./results/plots/accuracy_vs_epsilon_adv_query-1-shot.pdf', legend_location='upper right')

    file_list_AS_5 = [
        './results/data/adv_train_natural_context_protonets_5_5.txt',
        './results/data/adv_train_support_context_protonets_5_5.txt',
        './results/data/adv_train_query_context_protonets_5_5.txt'
    ]

    # plot_curves(file_list=file_list_AS_5, labels=labels, colors=colors, line_styles=line_styles, markers=markers,
    #             output_file='./results/plots/accuracy_vs_epsilon_adv_support-5-shot.pdf', legend_location='upper right')

    file_list_AQ_5 = [
        './results/data/adv_train_natural_target_protonets_5_5.txt',
        './results/data/adv_train_support_target_protonets_5_5.txt',
        './results/data/adv_train_query_target_protonets_5_5.txt'
    ]

    # plot_curves(file_list=file_list_AQ_5, labels=labels, colors=colors, line_styles=line_styles, markers=markers,
    #             output_file='./results/plots/accuracy_vs_epsilon_adv_query-5-shot.pdf', legend_location='upper right')

    file_lists = [
        file_list_AS_1,
        file_list_AQ_1,
        file_list_AS_5,
        file_list_AQ_5
    ]

    titles = [
        'Support Attack - 1-shot',
        'Query Attack - 1-shot',
        'Support Attack - 5-shot',
        'Query Attack - 5-shot'
    ]

    fig, axs = plt.subplots(2, 2, sharex=True, sharey=True)

    for file_list, title, ax in zip(file_lists, titles, [axs[0, 0], axs[0, 1], axs[1, 0], axs[1, 1]]):
        for file, label, color, line_style, marker in zip(file_list, labels, colors, line_styles, markers):
            data = np.genfromtxt(file, delimiter=',', names=['Epsilon', 'Accuracy', "Error"])
            ax.plot(data['Epsilon'], data['Accuracy'], color=color, label=label, linestyle=line_style, marker=marker)
            # ax.fill_between(data['Epsilon'], data['Accuracy'] - data['Error'],
            #                  data['Accuracy'] + data['Error'], color=color, alpha=0.2)
            ax.set_title(title)
            ax.legend(loc='upper right', fontsize='small')

        axs[1,0].set_xlabel('Epsilon', fontsize='x-large')
        axs[1,1].set_xlabel('Epsilon', fontsize='x-large')
        axs[0, 0].set_ylabel('Accuracy (%)', fontsize='x-large')
        axs[1, 0].set_ylabel('Accuracy (%)', fontsize='x-large')
    plt.savefig('./results/plots/accuracy_vs_epsilon_adv.pdf', bbox_inches='tight')
    plt.close()


if __name__ == '__main__':
    main()
