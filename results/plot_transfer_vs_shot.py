import matplotlib.pyplot as plt
import numpy as np

legend_labels = ['CNAPs to ResNet18', 'CNAPs to MNASNet']
colors = ['#5BB381', '#FFD700', '#DE3163', '#64C3EB']
line_styles = ['-', '--', '-.', ':']


def normalize_data(files):
    resnet_clean = np.genfromtxt(files[0], delimiter=',')
    resnet_general = np.genfromtxt(files[1], delimiter=',')
    mnasnet_clean = np.genfromtxt(files[2], delimiter=',')
    mnasnet_general = np.genfromtxt(files[3], delimiter=',')

    data = [
        100.0 * (resnet_clean - resnet_general) / resnet_clean,
        100.0 * (mnasnet_clean - mnasnet_general) / mnasnet_clean
    ]

    return data


def plot(ax, data, title):
    for i in range(len(data)):
        x = np.linspace(start = 1.0, stop=5.0, num=5) / 5.0
        ax.plot(x, data[i], color=colors[i], label=legend_labels[i], linestyle=line_styles[i])

    right_side = ax.spines["right"]
    right_side.set_visible(False)
    top = ax.spines["top"]
    top.set_visible(False)
    ax.set_xlabel('% Adversarial Examples', fontsize='large')
    ax.set_title(title, y=-0.25, fontsize='x-large', color='blue')


def main():
    files1 = [
        './data/transfer_cnaps_5_1_epsilon-05_resnet_clean.txt',
        './data/transfer_cnaps_5_1_epsilon-05_resnet_general.txt',
        './data/transfer_cnaps_5_1_epsilon-05_mnasnet_clean.txt',
        './data/transfer_cnaps_5_1_epsilon-05_mnasnet_general.txt'
    ]

    files2 = [
        './data/transfer_cnaps_5_1_epsilon-10_resnet_clean.txt',
        './data/transfer_cnaps_5_1_epsilon-10_resnet_general.txt',
        './data/transfer_cnaps_5_1_epsilon-10_mnasnet_clean.txt',
        './data/transfer_cnaps_5_1_epsilon-10_mnasnet_general.txt'
    ]

    titles = [
        'Transfer Attack (epsilon = 0.05)',
        'Transfer Attack (epsilon = 0.1)',
    ]

    fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(8,5))
    fig.set_dpi(300)

    for files, title, ax in zip([files1, files2], titles, [ax1, ax2]):
        data = normalize_data(files)
        plot(ax, data, title)

    ax1.legend(ncol=2, bbox_to_anchor=(0, 1), loc='lower left', fontsize='x-large')

    fig.text(-0.005, 0.6, 'Decrease in Accuracy (%)', va='center', rotation='vertical', fontsize='x-large')
    fig.tight_layout()

    plt.subplots_adjust(wspace=0.1)
    plt.savefig('./plots/transfer-cnaps_5_1.pdf', bbox_inches='tight')


if __name__ == '__main__':
    main()
