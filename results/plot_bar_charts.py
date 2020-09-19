import matplotlib.pyplot as plt
import numpy as np

x_labels = ['ProtoNets', 'MAML']
legend_labels = ['PGD Specific', 'PGD General', 'UAP', 'Noise', 'Swap', 'Label Shift']
colors = ['#5BB381', '#FFD700', '#DE3163', '#64C3EB', 'orange', 'purple']


def autolabel(ax, rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')


def plot(ax, file, title):
    data = np.genfromtxt(file, delimiter=',', skip_header=1, usecols=(1, 2))

    x = np.arange(len(x_labels))  # the label locations
    width = 0.15  # the width of the bars

    rects1 = ax.bar(x - width * 5 / 2, data[3], width, label=legend_labels[3], color=colors[0])
    rects2 = ax.bar(x - width * 3 / 2 , data[2], width, label=legend_labels[2], color=colors[1])
    rects3 = ax.bar(x - width / 2, data[5], width, label=legend_labels[5], color=colors[5])
    rects4 = ax.bar(x + width / 2, data[0], width, label=legend_labels[0], color=colors[2])
    rects5 = ax.bar(x + width * 3 / 2, data[1], width, label=legend_labels[1], color=colors[3])
    rects6 = ax.bar(x + width * 5 / 2, data[4], width, label=legend_labels[4], color=colors[4])

    right_side = ax.spines["right"]
    right_side.set_visible(False)
    top = ax.spines["top"]
    top.set_visible(False)

    ax.set_title(title, y=-0.20, fontsize='x-large')
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, fontsize='x-large')

    autolabel(ax, rects1)
    autolabel(ax, rects2)
    autolabel(ax, rects3)
    autolabel(ax, rects4)
    autolabel(ax, rects5)
    autolabel(ax, rects6)


def main():
    files = [
        './data/basic_support_5-way_1-shot_epsilon-05.txt',
        './data/basic_support_5-way_1-shot_epsilon-10.txt',
        './data/basic_support_5-way_5-shot_epsilon-05.txt',
        './data/basic_support_5-way_5-shot_epsilon-10.txt'
    ]

    titles = [
        'shot: 1, epsilon: 0.05',
        'shot: 1, epsilon: 0.1',
        'shot: 5, epsilon: 0.05',
        'shot: 5, epsilon: 0.1',
    ]

    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, sharey=True, figsize=(25,5))
    fig.set_dpi(300)

    for file, title, ax in zip(files, titles, [ax1, ax2, ax3, ax4]):
        plot(ax, file, title)

    ax1.legend(ncol=len(legend_labels), bbox_to_anchor=(0, 1), loc='lower left', fontsize='x-large')

    fig.text(-0.001, 0.5, 'Decrease in Accuracy (%)', va='center', rotation='vertical', fontsize='x-large')
    fig.tight_layout()

    plt.subplots_adjust(wspace=0.02)
    plt.savefig('./plots/basic.pdf', bbox_inches='tight')


if __name__ == '__main__':
    main()
