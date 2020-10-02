import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import griddata


def heatmap(data, row_labels, col_labels, ax=None,
            cbar_kw={}, cbarlabel="", show_xticks=True, **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (N, M).
    row_labels
        A list or array of length N with the labels for the rows.
    col_labels
        A list or array of length M with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if not ax:
        ax = plt.gca()

    # Plot the heatmap
    im = ax.imshow(data, vmin=0, vmax=100, **kwargs)

    # Create colorbar
    # cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    # cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # We want to show all ticks...
    if show_xticks:
        ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    # ... and label them with the respective list entries.
    if show_xticks:
        ax.set_xticklabels(col_labels, fontsize='x-large')
    ax.set_yticklabels(row_labels, fontsize='x-large')

    # Let the horizontal axes labeling appear on top.
    if show_xticks:
        ax.tick_params(top=True, bottom=False,
                       labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    if show_xticks:
        plt.setp(ax.get_xticklabels(), rotation=0, ha="center", rotation_mode="anchor")

    # Turn spines off and create white grid.
    for edge, spine in ax.spines.items():
        spine.set_visible(False)

    if show_xticks:
        ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im #, cbar


def annotate_heatmap(im, data1=None, data2=None, valfmt="{x:.2f}",
                     textcolors=["black", "white"],
                     threshold=None, **textkw):
    """
    A function to annotate a heatmap.

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A list or array of two color specifications.  The first is used for
        values below a threshold, the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """

    if not isinstance(data1, (list, np.ndarray)):
        data1 = im.get_array()

    if not isinstance(data2, (list, np.ndarray)):
        data2 = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data1.max())/2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data1.shape[0]):
        for j in range(data1.shape[1]):
            kw.update(color=textcolors[int(im.norm(data1[i, j]) > threshold)])
            # text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            text = im.axes.text(j, i, '{:.1f}'.format(data1[i, j]), **kw)
            texts.append(text)

    return texts


def main():
    fig, axs = plt.subplots(1, 2, sharey=True, sharex=True)

    class_labels = ["1", "3", "5"]
    shot_labels = ["1", "3", "5"]

    files = [
        './data/scale_maml_5-way_5-shot_epsilon-05.txt',
        './data/scale_cnaps_5-way_5-shot_epsilon-05.txt'
    ]

    titles = [
        'MAML, 5-shot (Clean 61.4%)',
        'CNAPs, 5-shot (Clean 66.7%)'
    ]

    images = []
    for file, title, ax in zip(files, titles, [axs[0], axs[1]]):
        data1 = np.genfromtxt(file, delimiter=',')

        im = heatmap(data1, shot_labels, class_labels, ax=ax, cmap="Reds", cbarlabel="% Drop in Accuracy")
        images.append(im)
        # texts = annotate_heatmap(im, valfmt="{x:.1f}", fontsize='x-large')
        texts = annotate_heatmap(im, data1=data1, data2=data1, valfmt="{x:.1f}", fontsize='x-large')
        ax.set_title(title, y=-0.15, fontsize='large', color='blue')
        ax.set_xlabel('Classes', fontsize='large')
        ax.xaxis.set_label_position('top')

    axs[0].set_ylabel('Shots', fontsize='large')
    plt.subplots_adjust(wspace=0.1)
    fig.tight_layout()
    plt.savefig('./plots/heat_maps_5-shot.pdf', bbox_inches='tight')

    fig, axs = plt.subplots(3, 1, sharex=True, sharey=True)

    class_labels = ["1", "2", "3", "4", "5"]
    shot_labels = ["1"]

    files = [
        './data/scale_protonets_5-way_1-shot_epsilon-05.txt',
        './data/scale_maml_5-way_1-shot_epsilon-05.txt',
        './data/scale_cnaps_5-way_1-shot_epsilon-05.txt'
    ]

    titles = [
        'ProtoNets, 1-shot (Clean 46.3%)',
        'MAML, 1-shot (Clean 45.9%)',
        'CNAPS, 1-shot (Clean 46.6%)'
    ]

    images = []
    for file, title, ax in zip(files, titles, [axs[0], axs[1], axs[2]]):
        data = np.genfromtxt(file, delimiter=',')
        data = np.expand_dims(data, axis=0)

        show_xticks = True
        # if ax == axs[1]:
        #     show_xticks = False

        im = heatmap(data, shot_labels, class_labels, ax=ax, cmap="Reds", cbarlabel="% Drop in Accuracy", show_xticks=show_xticks)
        images.append(im)
        texts = annotate_heatmap(im, valfmt="{x:.1f}", fontsize='x-large')
        ax.set_title(title, y=-0.45, fontsize='large', color='blue')
        ax.set_ylabel('Shots', fontsize='large')

    axs[0].xaxis.set_label_position('top')
    axs[0].set_xlabel('Classes', fontsize='large')
    plt.subplots_adjust(wspace=0.1)
    fig.tight_layout()
    plt.savefig('./plots/heat_maps_1-shot.pdf', bbox_inches='tight')


if __name__ == '__main__':
    main()
