import math
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt


def prep_colors_for_bars(df, labelCol, cmap=None, colors=None):
    compareOrder = sorted(list(df[labelCol].unique()))
    if cmap is None and colors is None:
        colors = list(sns.color_palette("tab10"))
    elif colors is None or not isinstance(colors, list):
        colors = [mpl.colors.rgb2hex(plt.get_cmap(cmap, len(compareOrder))(i)) for i in range(len(compareOrder))]
    else:
        colors *= math.ceil(len(compareOrder) / len(colors))
    return colors, compareOrder
