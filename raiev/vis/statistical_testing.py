from itertools import combinations
from scipy.stats import ks_2samp, mannwhitneyu, wasserstein_distance
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def stat_data(df, compareCol=None, confidenceCol="confidence", round_confidence=None):
    """
    Compute point density function and cumulative density function based on confidence.
    Optionally supply a column to split the data on to compute PDF and CDF separately.
    Return dataframes with PDF and CDF values for specified split or entire input dataframe

    :param df: (DataFrame) prediction data
    :param compareCol: (str) optional, column to split on before computation (if none, compute over entire dataframe)
    :param confidenceCol: (str) column name containing confidence values
    :param round_confidence: (int) optional, amount to round confidence to before computation
    """
    # Confirm provided column(s) are present in dataframe
    assert compareCol is None or compareCol in df.columns, f'Error! "{compareCol}" not in dataframe provided.'
    assert confidenceCol in df.columns, f'Error! "{confidenceCol}" not in dataframe provided.'

    pdf = df.copy()

    # If compareCol is None, compute PDF and CDF over entire dataframe (create temp col)
    if compareCol is None:
        noSplit, compareCol = True, "NoSplitTempCol"
        pdf[compareCol] = 0
    else:
        noSplit = False
    splitOrder = sorted(list(pdf[compareCol].unique()))

    # Round confidence if specified
    if round_confidence is not None and isinstance(round_confidence, int):
        pdf[confidenceCol] = pdf[confidenceCol].apply(lambda x: round(x, round_confidence))

    pdf = pdf.assign(n=1).groupby([compareCol, confidenceCol], as_index=False)[["n"]].sum()

    # Compute PDF and CDF
    data = {}
    for value in splitOrder:
        curr = pdf[pdf[compareCol] == value].sort_values(by=confidenceCol, ascending=True)
        curr["pdf"] = curr["n"] / curr["n"].sum()
        curr["cdf"] = curr["pdf"].cumsum()
        if noSplit:
            curr = curr.drop(columns=[compareCol])
        data[value] = curr

    return data


def stat_table(
    pdf, compareCol, confidenceCol="confidence", stat="mw", compareOrder=None, text="", add_title=None, rounding=4, alternative="less"
):
    stat = stat if stat in ["mw", "ks-pdf", "ks-cdf", "wd-pdf", "wd-cdf"] else "mw"
    alternative = alternative if alternative in ['less', 'greater', 'two-sided'] else 'less'
    alt_str = {'less':'<', 'greater':'>', 'two-sided': '='}[alternative]

    # Use provided compareOrder or create standard ordering
    if compareOrder is None or (isinstance(compareOrder, list) and len(compareOrder != 2)):
        if stat == "mw":
            compareOrder = sorted(list(pdf[compareCol].unique()))
        else:
            compareOrder = sorted(list(pdf.keys()))

    assert isinstance(compareOrder, list) and len(compareOrder) == 2, "Error! provided data must "

    # Computations for each stat type
    calc = {
        "mw": lambda x, y: mannwhitneyu(
            pdf[pdf[compareCol] == x][confidenceCol],
            pdf[pdf[compareCol] == y][confidenceCol],
            alternative=alternative,
            method="auto",
        ),
        "ks-pdf": lambda x, y: ks_2samp(pdf[x]["pdf"], pdf[y]["pdf"], alternative=alternative, method="asymp"),
        "ks-cdf": lambda x, y: ks_2samp(pdf[x]["cdf"], pdf[y]["cdf"], alternative=alternative, method="asymp"),
        "wd-pdf": lambda x, y: wasserstein_distance(pdf[x]["pdf"], pdf[y]["pdf"]),
        "wd-cdf": lambda x, y: wasserstein_distance(pdf[x]["cdf"], pdf[y]["cdf"]),
    }[stat]

    c1, c2 = calc(compareOrder[0], compareOrder[1]), calc(compareOrder[1], compareOrder[0])

    if "wd" in stat:
        stat_df = pd.DataFrame(
            {"distance": [c1, c1]},
            index=[f"{text}{compareOrder[0]} {alt_str} {compareOrder[1]}", f"{text}{compareOrder[1]} {alt_str} {compareOrder[0]}"],
        )
    else:
        stat_df = pd.DataFrame(
            {"pvalue": [c1.pvalue, c2.pvalue], "statistic": [c1.statistic, c2.statistic]},
            index=[f"{text}{compareOrder[0]} {alt_str} {compareOrder[1]}", f"{text}{compareOrder[1]} {alt_str} {compareOrder[0]}"],
        )

    if add_title is not None:
        title = {
            "mw": "Mann-Whitney U",
            "ks-pdf": "Kolmogorov-Smirnov (PDF)",
            "ks-cdf": "Kolmogorov-Smirnov (CDF)",
            "wd-pdf": "Wasserstein (PDF)",
            "wd-cdf": "Wasserstein (CDF)",
        }[stat]
        title = f"{add_title} {title}"
        stat_df = stat_df.rename_axis(title, axis=1)

    if rounding is not None and isinstance(rounding, int):
        stat_df = stat_df.round(rounding)

    return stat_df


def stat_compare(predictions, compareCol, confidenceCol="confidence", stats="all", filter_rank=None):
    """
    Compute point density function and cumulative density function based on confidence.

    :param predictions: (DataFrame) prediction data
    :param compareCol: (str) column name containing values to make comparisons with
    :param confidenceCol: (str) column name containing confidence values
    :param stats: (str or list) can supply a list of multiple, a string as a single stat, or keyword 'all' to run all stats.
                         choices=['mw', 'ks-pdf', 'ks-cdf', 'wd-pdf', 'wd-cdf']
    :return: (DataFrame) results from statistical testing and dataframe with PDF / CDF values
    """
    assert confidenceCol in predictions.columns, f'Error! "{confidenceCol}" not in dataframe provided.'
    assert compareCol in predictions.columns, f'Error! "{compareCol}" not in dataframe provided.'

    # Load in stats
    avail = ["mw", "ks-pdf", "ks-cdf", "wd-pdf", "wd-cdf"]
    if stats == "all":
        stats = avail
    if isinstance(stats, str) and stats in avail:
        stats = [stats]
    stats = [stat for stat in stats if stat in avail]
    assert isinstance(stats, list) and len(stats) > 0, "Error! Please enter valid statistic"

    compareValues = list(predictions[compareCol].unique())

    pdf_data = {}
    for value in compareValues:
        df = predictions[predictions[compareCol] == value]
        # stat_data returns a dictionary with a key value pair for the selected value
        pdf_data.update(stat_data(df, compareCol=compareCol, confidenceCol=confidenceCol))

    # Compute all combinations of elements in compareCol
    pairs = list(combinations(sorted(compareValues), 2))

    # Build dataframe for each statistic
    stat_df = []
    for stat in stats:
        stat_name = {
            "mw": "Mann-Whitney U",
            "ks-pdf": "Kolmogorov-Smirnov (PDF)",
            "ks-cdf": "Kolmogorov-Smirnov (CDF)",
            "wd-pdf": "Wasserstein (PDF)",
            "wd-cdf": "Wasserstein (CDF)",
        }[stat]

        dfs = []
        for pair in pairs:
            if stat == "mw":
                data = predictions[predictions[compareCol].apply(lambda x: x in pair)]
                p1, p2 = len(data[data[compareCol] == pair[0]]), len(data[data[compareCol] == pair[1]])

            else:
                data = {}
                data[pair[0]] = pdf_data[pair[0]]
                data[pair[1]] = pdf_data[pair[1]]
                p1, p2 = len(data[pair[0]]), len(data[pair[0]])

            if p1 > 19 and p2 > 19:
                dfs.append(
                    stat_table(
                        data, compareCol, confidenceCol=confidenceCol, stat=stat, add_title=False, rounding=None
                    )
                )

        df = pd.concat(dfs)

        # Add "rank" column
        if stat in ["ks-pdf", "ks-cdf", "mw"]:
            # For ks and mw stats rank is based on pvalue and statistic
            df = df.sort_values(by=["pvalue", "statistic"], ascending=[True, False])
            df["rank"] = range(1, len(df) + 1)
            if filter_rank is not None and isinstance(filter_rank, float):
                df["rank"] = df.apply(lambda x: -1 if x["pvalue"] > filter_rank else int(x["rank"]), axis=1)

            # Rename columns to include the statistic
            df = df.rename(
                columns={"pvalue": f"{stat_name} pval", "statistic": f"{stat_name} stat", "rank": f"{stat_name} rank"}
            )
        else:
            # For wd stats the rank is based on distance
            # pair <a, b> and pair <b, a> have the same rank
            # Code below groups pairs and applies rank before exploding to full set
            df = df.reset_index()
            df["temp"] = [" < ".join(sorted(i.split(" < "))) for i in df["index"]]
            df = (
                df.groupby("temp")
                .aggregate({"distance": "first", "index": lambda x: list(x)})
                .sort_values(by="distance", ascending=False)
            )
            df[f"{stat_name} rank"] = range(1, len(df) + 1)
            df = df.rename(columns={"distance": f"{stat_name} stat"})
            df = df.explode("index").set_index("index").rename_axis(None, axis=0)

        stat_df.append(df)

    stat_df = pd.concat(stat_df, axis=1).reset_index().rename(columns={"index": "comparison"})
    stat_df["dist1"] = stat_df["comparison"].apply(lambda x: x.split(" < ")[0])
    stat_df["dist2"] = stat_df["comparison"].apply(lambda x: x.split(" < ")[1])
    stat_df["comparisonCol"] = compareCol
    stat_df["confidenceCol"] = confidenceCol

    return stat_df, pdf_data


def stat_plot(
    pdf,
    confidenceCol="confidence",
    confidenceName="Confidence",
    dist="cdf",
    compareOrder=None,
    compareColor=None,
    compareLine=None,
    title="",
    ax=None,
    show_labels=True,
    legend=True,
):
    dist_dict = {"cdf": "Cumulative Density", "pdf": "Probability Density"}

    if compareOrder is None:
        compareOrder = sorted(list(pdf.keys()))
    if compareColor is None:
        compareColor = list(sns.color_palette("tab10"))[: len(compareOrder)]
    if compareLine is None:
        compareLine = ["-"] * len(compareOrder)

    if len(compareColor) < len(compareOrder):
        compareColor += list(sns.color_palette("tab10"))[: (len(compareOrder) - len(compareColor))]
    if len(compareLine) < len(compareOrder):
        compareLine += ["-"] * (len(compareOrder) - len(compareLine))

    if ax == None:
        fig, ax = plt.subplots(1, 1, sharey=True, sharex=True, figsize=(5, 2.5))

    for value, color, line in zip(compareOrder, compareColor, compareLine):
        try:
            ax.plot(pdf[value][confidenceCol], pdf[value][dist], label=value, color=color, linestyle=line)
        except KeyError:
            pass

    ax.set_title(title, fontsize=11, fontname="serif", style="italic")
    ax.spines.top.set(visible=False)
    ax.spines.right.set(visible=False)

    ax.set_xlabel('Prediction Confidence', fontname="serif")
    ax.set_ylabel(dist_dict[dist], fontsize=11, fontname="serif")

    if show_labels:
        ax.set_xticks(ax.get_xticks())
        ax.set_xticklabels(ax.get_xticklabels(), font="serif")

        ax.set_yticks(ax.get_yticks())
        ax.set_yticklabels(ax.get_yticklabels(), font="serif")

    if legend:
        ax.legend(bbox_to_anchor=[1, 1], prop={"family": "serif"}, frameon=False)
