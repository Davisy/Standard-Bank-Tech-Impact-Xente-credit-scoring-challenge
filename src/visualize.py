# import important packages
from imports import *


# set colors
dark_colors = [
    "#A51C30",
    "#808080",
    (0.8509803921568627, 0.37254901960784315, 0.00784313725490196),
    (0.4588235294117647, 0.4392156862745098, 0.7019607843137254),
    (0.9058823529411765, 0.1607843137254902, 0.5411764705882353),
    (0.4, 0.6509803921568628, 0.11764705882352941),
    (0.9019607843137255, 0.6705882352941176, 0.00784313725490196),
    (0.6509803921568628, 0.4627450980392157, 0.11372549019607843),
    (0.4, 0.4, 0.4),
]
SPINE_COLOR = "black"


# A function to draw confusion matrix plot
def plot_confusion_matrix(
    y_true, y_pred, classes, title="Confusion matrix", cmap=plt.cm.Blues, fig_num=None
):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """

    if fig_num is not None:
        plt.subplot(2, 2, fig_num)
    fmt = "d"
    cm = confusion_matrix(y_true, y_pred)
    plt.imshow(cm, interpolation="nearest", cmap=cmap)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)
    plt.title("")

    thresh = cm.max() / 2.0
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(
            j,
            i,
            format(cm[i, j], fmt),
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black",
        )
    plt.ylabel("True label")
    plt.xlabel("Predicted label")


# a function to save the confusion matrix plot
def savefig(filename, leg=None, format=".pdf", *args, **kwargs):
    """
    Save in PDF file with the given filename.
    """
    if leg:
        art = [leg]
        plt.savefig(
            filename + format,
            additional_artists=art,
            bbox_inches="tight",
            *args,
            **kwargs
        )
    else:
        plt.savefig(filename + format, bbox_inches="tight", *args, **kwargs)
    plt.close()
