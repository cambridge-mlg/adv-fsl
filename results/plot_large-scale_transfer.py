import matplotlib.pyplot as plt
import numpy as np

x_labels = ['ILSVRC', 'Aircraft', 'Birds', 'Quick Draw', 'VGG Flower', 'Traffic Signs', 'MSCOCO', 'MNIST', 'CIFAR10', 'CIFAR100']
legend_labels = ['ResNet18', 'MNASNet']
colors = colors = ['#DE3163', '#64C3EB']


def autolabel(ax, rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 1),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', rotation=90, fontsize='x-small')


def plot(ax, file, title):
    data = np.genfromtxt(file, delimiter=',', skip_header=1, usecols=(1, 3, 4, 6, 8, 9, 10, 11, 12, 13))

    x = np.arange(len(x_labels))  # the label locations
    width = 0.18  # the width of the bars

    rects1 = ax.bar(x - width * 2, data[0], width, label=legend_labels[0], color=colors[0])
    rects2 = ax.bar(x + width * 2, data[1], width, label=legend_labels[1], color=colors[1])

    right_side = ax.spines["right"]
    right_side.set_visible(False)
    top = ax.spines["top"]
    top.set_visible(False)

    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, fontsize='medium', rotation=20)

    autolabel(ax, rects1)
    autolabel(ax, rects2)


def main():
    files = [
        './data/large_scale_transfer.txt'
    ]

    fig, axs = plt.subplots(sharey=True, figsize=(8, 3))
    fig.set_dpi(300)

    for file, ax in zip(files, [axs]):
        plot(ax, file, None)

    axs.legend(ncol=len(legend_labels), bbox_to_anchor=(0, 1.1), loc='lower left', fontsize='medium')

    fig.text(-0.002, 0.5, 'Decrease in Accuracy (%)', va='center', rotation='vertical', fontsize='medium')
    fig.tight_layout()

    plt.subplots_adjust(wspace=0.05)
    plt.savefig('./plots/large-scale_transfer.pdf', bbox_inches='tight')


if __name__ == '__main__':
    main()
