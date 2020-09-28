import matplotlib.pyplot as plt
import numpy as np


legend_labels = ['ResNet18', 'MNASNet']
colors = ['#DE3163', '#64C3EB']


def autolabel(ax, rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')


def plot(ax, file, title, x_labels):
    data = np.genfromtxt(file, delimiter=',', skip_header=1, usecols=(1, 2))

    x = np.arange(len(x_labels))  # the label locations
    width = 0.2  # the width of the bars

    rects1 = ax.bar(x - width / 2, data[0], width, label=legend_labels[0], color=colors[0])
    rects2 = ax.bar(x + width / 2, data[1], width, label=legend_labels[1], color=colors[1])

    right_side = ax.spines["right"]
    right_side.set_visible(False)
    top = ax.spines["top"]
    top.set_visible(False)

    ax.set_title(title, y=-0.4, fontsize='x-large', color='blue')
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, fontsize='x-large')

    autolabel(ax, rects1)
    autolabel(ax, rects2)


def main():
    files = [
        './data/small-scale_transfer_protonets.txt',
        './data/small-scale_transfer_maml.txt',
        # './data/small-scale_transfer_cnaps.txt'
    ]

    titles = [
        'ProtoNets',
        'MAML',
        # '(c) CNAPs'
    ]

    x_labels_list = [
        ['1-shot', '5-shot'],
        ['1-shot', '5-shot'],
        # ['1-shot', '5-shot']
    ]

    fig, axs = plt.subplots(1, 2, sharey=True, figsize=(5.0,3.5))
    fig.set_dpi(300)

    for file, title, ax, x_labels in zip(files, titles, [axs[0], axs[1]], x_labels_list):
        plot(ax, file, title, x_labels)

    axs[0].legend(ncol=len(legend_labels), bbox_to_anchor=(0, 1.1), loc='lower left', fontsize='x-large')

    fig.text(-0.002, 0.55, 'Decrease in Accuracy (%)', va='center', rotation='vertical', fontsize='large')
    fig.tight_layout()

    plt.subplots_adjust(wspace=0.05)
    plt.savefig('./plots/small-scale_transfer.pdf', bbox_inches='tight')


if __name__ == '__main__':
    main()
