import math
import IPython
import pandas as pd
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
from raiev.assistance import characterizeClusters, characterizeClusters_LLM


def add_high_confidence(predictions, threshold=0.9, upper=None, confidenceCol="confidence"):
    """
    Add boolean column "high confidence" to denote if confidence is high or not

    :param predictions: (Pandas DataFrame) containing 'confidence' column
    :param threshold: (float) lower bound threshold (inclusive)
    :param upper: (float) optional, upper bound threshold (inclusive)
    """
    assert confidenceCol in list(predictions.columns), f"Error: {confidenceCol} column must be in dataframe supplied"
    upper = predictions[confidenceCol].max() if upper is None or upper <= threshold else upper
    predictions["high confidence"] = predictions[confidenceCol].apply(lambda x: x >= threshold and x <= upper)
    return predictions


def plot_kde(
        dfs,
        confidenceCol="confidence",
        label=None,
        title="",
        colors=list(sns.color_palette("tab10")),
        lines=None,
        fig_kwargs={},
        xlim=None,
        ylim=None,
        legend=True,
):
    """
    Create a single KDE plot. DataFrame(s) supplied must contain 'confidence' column.

    :param dfs: (list or Pandas DataFrame)
    :param confidenceCol: (str)
    :param label: (list or str)
    :param title: (str)
    :param colors: (list of colors)
    :param lines: (list of line styles)
    :param fig_kwargs: (dict)
    """
    dfs = [dfs] if type(dfs) == pd.core.frame.DataFrame else dfs
    assert type(dfs) == list, "Error: Must supply a Pandas DataFrame or list of Pandas DataFrames"

    label = [""] * len(dfs) if label is None else label
    label = label if type(label) == list else [label]
    colors *= math.ceil(len(dfs) / len(colors))
    lines = ["-"] * len(dfs) if lines is None else lines
    lines += ["-"] * (len(dfs) - len(lines))

    fig_kw = {"figsize": (5, 3)}
    fig_kw.update(fig_kwargs)
    fig, ax = plt.subplots(1, 1, **fig_kw)
    for i, df in enumerate(dfs):
        plot_kwargs = {"ax": ax, "label": label[i], "color": colors[i], "linestyle": lines[i]}
        sns.kdeplot(list(df[confidenceCol]), **plot_kwargs)

    plt.xlabel("Prediction Confidence", fontname="serif")
    plt.ylabel("KDE", fontname="serif")
    plt.xticks(fontname="serif")
    plt.yticks(fontname="serif")
    if xlim is not None:
        plt.xlim(xlim)
    if ylim is not None:
        plt.ylim(ylim)
    if legend:
        plt.legend(bbox_to_anchor=[1, 1], prop={"family": "serif"}, frameon=False)
    plt.title(title, fontsize=11, fontname="serif", style="italic")
    plt.show()


def compare_across(
        df,
        confidenceCol="confidence",
        compareCol=None,
        compareOrder=None,
        overall=True,
        title="",
        colors=list(sns.color_palette("tab10")),
        lines=None,
        fig_kwargs={},
        xlim=None,
        ylim=None,
        legend=True,
):
    """
    Create KDE plot with DataFrame split on the compareCol supplied.

    :param df: (Pandas DataFrame)
    :param confidenceCol: (str)
    :param compareCol: (str)
    :param compareOrder: (list)
    :param overall: (bool)
    :param title: (str)
    :param colors: (list of colors)
    :param lines: (list of linestyles)
    :param fig_kwargs: (dict)
    """
    dfs, labels = [], []
    if overall:
        dfs.append(df.copy())
        labels.append("Overall")
    if compareCol is not None:
        if compareOrder is None:
            compareOrder = sorted(df[compareCol].unique())
        for value in compareOrder:
            dfs.append(df[df[compareCol] == value].copy())
            labels.append(value)

    plot_kde(
        dfs,
        confidenceCol=confidenceCol,
        label=labels,
        title=title,
        colors=colors,
        lines=lines,
        fig_kwargs=fig_kwargs,
        xlim=xlim,
        ylim=ylim,
        legend=legend,
    )


def error_data(df, labelCol, plot_type="error", correctCol="correct", highConfidenceCol='high confidence',
               compareOrder=None, perModel=False, colors=list(sns.color_palette("tab10")), overall=False, sort=False):
    """
    Computes error rate / high confidence rate percentages based plot type supplied:

        plot_type = error: computes high confidence errors over total errors

        plot_type = total: computes percent high confidence errors and percent total errors (stored in 'y' as a list)

    :param df: (Pandas DataFrame)
    :param labelCol: (str)
    :param plot_type: (str) choices: error, total
    :param correctCol: (str)
    :param highConfidenceCol: (str)
    :param compareOrder: (list) optional, must include unique values in the labelCol column
    :param perModel: (bool)
    :param colors: (list) optional
    :param overall: (bool) optional, for 'error' plots add an overall bar
    :param sort: (bool) optional, to sort bars based on "y" value
    :return: new dataframe with bar plotting data: label, x, y, and color
    """
    assert plot_type in ["error", "total"], "Please supply a valid plot_type [error, total]"

    compareOrder = compareOrder if compareOrder is not None else sorted(list(df[labelCol].unique()))
    colors *= math.ceil(len(compareOrder) / len(colors))

    if df[correctCol].dtype == int:
        df[correctCol] = df[correctCol].astype(bool)

    total = len(df)
    incorrect_df = df[~df[correctCol]]

    bars = []
    for label, color in zip(compareOrder, colors):
        incorrect_label_df = incorrect_df[incorrect_df[labelCol] == label]
        error = len(incorrect_label_df)

        if perModel:
            total = len(df[df[labelCol] == label])
            if total == 0:
                continue

        high_confidence = len(incorrect_label_df[incorrect_label_df[highConfidenceCol]])

        if plot_type == 'error':
            high_confidence_over_error = ((high_confidence / error) * 100) if error > 0 else 0
            bars.append((label, color, high_confidence_over_error, high_confidence, error, total))
        else:
            error_over_total = (error / total) * 100
            high_confidence_over_total = (high_confidence / total) * 100
            bars.append((label, color, [error_over_total, high_confidence_over_total], high_confidence, error, total))

    output = pd.DataFrame(bars, columns=[labelCol, 'color', 'y', "High Confidence", "Error", "Total"])

    if sort:
        output = output.sort_values(by='y', ascending=False).reset_index(drop=True)
        compareOrder = list(output[labelCol].unique())

    if plot_type == 'error' and overall:
        error = len(incorrect_df)
        high_confidence = len(incorrect_df[incorrect_df[highConfidenceCol]])
        high_confidence_over_error = ((high_confidence / error) * 100) if error > 0 else 0
        output = pd.concat(
            [pd.DataFrame({labelCol: ['Overall'], 'y': [high_confidence_over_error], 'color': ['black'],
                           "High Confidence": high_confidence, "Error": error, "Total": total}), output])
        compareOrder = ['Overall'] + compareOrder

    output = output.reset_index(drop=True).reset_index().rename(columns={'index': 'x'})
    return output, compareOrder


def plot_error_bars(
        data,
        labelCol,
        plot_type="error",
        compareOrder=None,
        title=None,
        ticklabels=True,
        annot=True,
        log=False,
        horizontal=False,
        descName=None,
        desc=None,
        clusterCol="model_alias",
        verbose=True
):
    """
    Plot error bars.

    :param data: (Pandas DataFrame)
    :param plot_type: (str) choices: error, total
    :param labelCol: (str)
    :param compareOrder: (list) optional, must include unique values in the labelCol column
    :param title: (str) optional
    :param ticklabels: (bool) optional
    :param annot: (bool) optional, only applied if plot_type == 'error'
    :param log: (bool) optional
    :param horizontal: (bool) optional
    :param descName:(str) optional
    :param desc: (dict) optional
    :param verbose: (bool) optional, default True. Prints natural language description of clusters.
    """
    assert plot_type in ["error", "total"], "Please supply a valid plot_type [error, total]"

    title = (
        {"error": "High Confidence Errors Relative to Total Errors",
         "total": "High Confidence Errors Relative to Total Predictions"
         }[plot_type] if title is None else title)

    if plot_type == "total":
        annot, overall = False, False

    plotDesc = None
    if verbose: 
        try:
            plotDesc = desc[descName]
        except KeyError:
            plotDesc = characterizeClusters_LLM(data, name=descName, modelCol=clusterCol)
        if plotDesc != "":
            IPython.display.display(
                IPython.display.HTML(
                    f"<h7>{plotDesc}</h7>"
                )
            )
    if horizontal:
        compareOrder = compareOrder[::-1]
        data = data[::-1].drop(columns=['x']).reset_index(drop=True).reset_index().rename(columns={'index': 'x'})

    figsize = (7, len(compareOrder) * 0.5) if horizontal else (len(compareOrder) * 2.5, 3)

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.spines.top.set(visible=False)
    ax.spines.right.set(visible=False)

    for _, row in data.iterrows():
        if plot_type == "error":
            if horizontal:
                ax.barh(row["x"], row["y"], align="edge", height=0.9, color=row["color"], label=row[labelCol], log=log)
            else:
                ax.bar(row["x"], row["y"], align="edge", width=0.9, color=row["color"], label=row[labelCol], log=log)
        else:
            if horizontal:
                ax.barh(row["x"], 100, align="edge", height=0.9, color="gainsboro", alpha=0.4, log=log)
                ax.barh(row["x"], row["y"][0], align="edge", height=0.9, color=row["color"], alpha=0.2, log=log)
                ax.barh(row["x"], row["y"][1], align="edge", height=0.9, color=row["color"], label=row[labelCol],
                        log=log)
            else:
                ax.bar(row["x"], 100, align="edge", width=0.9, color="gainsboro", alpha=0.4, log=log)
                ax.bar(row["x"], row["y"][0], align="edge", width=0.9, color=row["color"], alpha=0.2, log=log)
                ax.bar(row["x"], row["y"][1], align="edge", width=0.9, color=row["color"], label=row[labelCol], log=log)

    if plot_type == "error" and not log:
        maximum = math.ceil(data["y"].max())
    elif plot_type != "error" and not log:
        maximum = math.ceil(data["y"].apply(lambda x: x[0]).max())
    else:
        maximum = 0

    if annot:
        for i, y in enumerate(list(data["y"])):
            if horizontal:
                ax.text(y, i + 0.4, f" {y:.2f}%", size=9, fontname="serif", ha="left", va="center")
            else:
                ax.text(i + 0.45, y, f"{y:.2f}%", size=9, fontname="serif", ha="center", va="bottom")

    if ticklabels:
        ticklabel_func = ax.set_yticks if horizontal else ax.set_xticks
        ha = 'right' if horizontal else 'center'
        ticklabel_func(
            [x + 0.45 for x in range(0, len(compareOrder))],
            compareOrder,
            size=9,
            fontname="serif",
            ha=ha,
            va="center",
        )
        plt.tick_params(left=False, labelleft=True) if horizontal else plt.tick_params(left=False, labelbottom=True,
                                                                                       bottom=False)
    else:
        ax.set_yticks([]) if horizontal else ax.set_xticks([])

    if log and horizontal:
        ticks = ax.get_xticks()
        ax.xaxis.set_ticks(ticks, ['{:.2}'.format(x) for x in ticks], visible=True, minor=True)

    metric = "%" if not log else "log"
    ylabel = "High Conf. Errors / Total Errors" if plot_type == "error" else "Error Rate"
    ylabel = f"{ylabel} ({metric})"  # append log or % to clarify
    ax.set_xlabel(ylabel, fontname="serif") if horizontal else ax.set_ylabel(ylabel, fontname="serif")

    if maximum > 0:
        ax.set_xlim([0, maximum]) if horizontal else ax.set_ylim([0, maximum])

    if title is not None:
        ax.set_title(title + "\n", fontsize=11, fontname="serif", style="italic")

    if horizontal:
        plt.margins(y=0)
    else:
        ax.legend(bbox_to_anchor=[1, 1], loc="upper left", prop={"family": "serif"}, frameon=False)

    plt.show()
    return plotDesc


def plot_clusters(df, grp_dict, model_col="model_alias", original_coords=False, hueCol="GroupCol"):
    """
    Visualize clusters.

    :param df: (Pandas DataFrame) data for plotting
    :param grp_dict: (dict) Dictionary of model to cluster mappings.
    :param model_col: (str) optional, default model_alias. Name of model columns
    :param original_coords: (boolean) optional, default False. Indicates plotting using original coordinates if 2D.
    :param hueCol: (str)
    """
    df = df.reset_index(drop=False)
    df["GroupCol"] = df[model_col].map(grp_dict)
    if original_coords:
        x_col = "x"
        y_col = "y"
    else:
        x_col = "dim0"
        y_col = "dim1"

    plt.figure()
    sns.scatterplot(data=df, x=x_col, y=y_col, hue=hueCol, palette='tab10')
    plt.show()
