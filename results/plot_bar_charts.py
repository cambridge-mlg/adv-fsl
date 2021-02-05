import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
import matplotlib.pyplot as plt
import numpy as np


legend_labels = ['ASP Specific', 'ASP General (10x)', 'ASP General (1x)', 'Noise', 'Swap', 'Label Shift']
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


def plot(ax, file, title, x_labels):
    data = np.genfromtxt(file, delimiter=',', skip_header=1, usecols=(1, 2))

    x = np.arange(len(x_labels))  # the label locations
    width = 0.15  # the width of the bars

    rects1 = ax.bar(x - width * 5 / 2, data[5], width, label=legend_labels[5], color=colors[5])
    rects2 = ax.bar(x - width * 3 / 2, data[3], width, label=legend_labels[3], color=colors[0])
    rects3 = ax.bar(x - width / 2 , data[0], width, label=legend_labels[0], color=colors[2])
    rects4 = ax.bar(x + width / 2, data[2], width, label=legend_labels[2], color=colors[1])
    rects5 = ax.bar(x + width * 3 / 2, data[1], width, label=legend_labels[1], color=colors[4])
    rects6 = ax.bar(x + width * 5 / 2, data[4], width, label=legend_labels[4], color=colors[3])

    right_side = ax.spines["right"]
    right_side.set_visible(False)
    top = ax.spines["top"]
    top.set_visible(False)

    ax.set_title(title, y=-0.32, fontsize='x-large', color='blue')
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
        './data/basic_support_5-way_1-shot_epsilon-0314.txt',
        './data/basic_support_5-way_1-shot_epsilon-05.txt',
        './data/basic_support_5-way_5-shot_epsilon-0314.txt',
        './data/basic_support_5-way_5-shot_epsilon-05.txt',
    ]

    titles = [
        '(a) shot: 1, epsilon: 0.0314',
        '(b) shot: 1, epsilon: 0.05',
        '(c) shot: 5, epsilon: 0.0314',
        '(d) shot: 5, epsilon: 0.05'
    ]

    x_labels_list = [
        ['ProtoNets (Clean 46.4%)', 'MAML (Clean 46.8%)'],
        ['ProtoNets (Clean 46.4%)', 'MAML (Clean 46.8%)'],
        ['ProtoNets (Clean 64.7%)', 'MAML (Clean 60.8%)'],
        ['ProtoNets (Clean 64.7%)', 'MAML (Clean 60.8%)']
    ]

    fig, axs = plt.subplots(2, 2, sharey=True, figsize=(12.5,5.5))
    fig.set_dpi(300)

    for file, title, ax, x_labels in zip(files, titles, [axs[0,0], axs[0,1], axs[1,0], axs[1,1]], x_labels_list):
        plot(ax, file, title, x_labels)

    axs[0,0].legend(ncol=len(legend_labels), bbox_to_anchor=(0, 1.05), loc='lower left', fontsize='large')

    fig.text(-0.002, 0.5, 'Relative Decrease in Accuracy (%)', va='center', rotation='vertical', fontsize='x-large')
    fig.tight_layout()

    plt.subplots_adjust(wspace=0.05)
    plt.savefig('./plots/basic.pdf', bbox_inches='tight')


if __name__ == '__main__':
    main()
