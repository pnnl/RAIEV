import IPython
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import math
from raiev.vis import stat_table, stat_data

from .base import RAIEVanalysis, cleanParams

from wordcloud import WordCloud
from wordcloud import ImageColorGenerator
import warnings

warnings.filterwarnings('ignore')

WORKFLOW_CATEGORY = "Performance Equity"
WORKFLOW_DESC = "Performance Equity evaluations provide insight on whether there is an equitable reliability and robustness across circumstances in which models perform (e.g., specific locations) as well as characteristic-based groupings of model inputs (e.g., variants of inputs like long vs short documents)."

WORKFLOW_OUTLINE = """## import the raiev package
import raiev
## import the Performance Equity workflow sub-module
from raiev.workflows import equity

## Load model predictions
predictions = load_predictions(filepath_to_predictions)
    
## Instantiation of workflow object
positiveLabel = 'Positive Class' 
negativeLabel = 'Negative Class'
pe = equity.analysis(predictions,positiveLabel=positiveLabel,negativeLabel=negativeLabel)

## Analyze distribution of model confidence values across analyst uncertainty bins
## for correct vs. incorrect predictions
pe.uncertaintyAcrossCorrectness()

## Analyze distribution of model confidence values across analyst uncertainty bins
## for high confidence errors
pe.uncertaintyForHighConfidenceErrors()

## Continued Analysis, Analytic Interface(s), etc. under development
"""

def workflow():
    """""" 
    IPython.display.display(IPython.display.HTML(f"<h3>Overview of {WORKFLOW_CATEGORY} Workflow</h3>"))
    print(WORKFLOW_DESC)
    IPython.display.display(IPython.display.HTML("<b>Implemented workflow includes:</b>"))
    print(WORKFLOW_OUTLINE)


UNCERTAINTY_RANKINGS = [
    ("almost no chance", 0.05),
    ("very unlikely", 0.2),
    ("unlikely", 0.45),
    ("roughly even chance", 0.55),
    ("likely", 0.8),
    ("very likely", 0.95),
    ("almost certainly", 1),
]
colors = list(sns.color_palette("tab20c")[4:8]) + list(sns.color_palette("tab20b"))[1:4][::-1]
color_dict = {n: c for (n, _), c in zip(UNCERTAINTY_RANKINGS, colors)}


def _UNCERTAINTY_RANKINGS_LEGEND():
    fontname, fontsize = "sans-serif", 20

    df = pd.DataFrame(UNCERTAINTY_RANKINGS, columns=["index", "max"])
    df["min"] = [0] + list(df["max"])[:-1]
    df["len"] = df.apply(lambda x: x["max"] - x["min"], axis=1)
    df = df[["index", "min", "max", "len"]].rename_axis(None, axis=0)
    df["color"] = colors

    fig, ax = plt.subplots(figsize=(12, 2))
    df.set_index("index").T.loc[["len"]].plot.barh(stacked=True, color=colors, ax=ax)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)

    ax.yaxis.set_ticks_position("none")
    ax.set_ylim([0, 0.5])
    ax.set_yticks([])

    xticks = [0] + list(df["max"])
    xticklabels = ["   0%", "      5%"] + [f"   {int(x * 100)}%" for x in xticks[2:-2]] + ["95%    ", "     100%"]
    ax.set_xticks(xticks, xticklabels, fontname=fontname, fontsize=fontsize)
    ax.xaxis.set_ticks_position("none")

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(
        handles[::-1],
        labels[::-1],
        bbox_to_anchor=[-0.04, 1],
        title="",
        frameon=False,
        prop={"family": fontname, "size": fontsize},
    )
    return fig


def _vertUncertaintyBars(predictions, correctCol, goldCol, predCol, positiveLabel, negativeLabel):
    fontname, fontsize = "sans-serif", 16
    if predictions[correctCol].dtype == int:
        predictions[correctCol] = predictions[correctCol].astype(bool)
    errors = predictions[~(predictions[correctCol])].copy()

    if positiveLabel not in predictions[goldCol].unique():
        print(f"{positiveLabel} not found in {goldCol}")
        return

    def label_false(x):
        if x[goldCol] == negativeLabel and x[predCol] == positiveLabel:
            return "False Positives"
        if x[goldCol] == positiveLabel and x[predCol] == negativeLabel:
            return "False Negatives"
        return ""

    errors["col"] = errors.apply(lambda x: label_false(x), axis=1)
    compareOrder = ["False Positives", "False Negatives"]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.spines.top.set(visible=False)
    ax.spines.left.set(visible=False)
    ax.spines.bottom.set(visible=False)
    maximum, percentages = 0, []
    for i, col in enumerate(compareOrder):
        curr = errors[errors["col"] == col]
        total = len(curr)
        very_likely = len(curr[curr["confidence ranking"] == "very likely"])
        almost_certainly = len(curr[curr["confidence ranking"] == "almost certainly"])
        if total == 0:
            perc = perc2 = 0
        else:
            perc = ((very_likely + almost_certainly) / total) * 100
            perc2 = (very_likely / total) * 100

        l1, l2 = (None, None) if i != 0 else ("almost certainly", "very likely")

        ax.bar(i, perc, align="edge", width=0.9, label=l1, color=colors[-1])
        ax.bar(i, perc2, align="edge", width=0.9, label=l2, color=colors[-2])
        maximum = perc if maximum < perc else maximum
        percentages.append(perc)

    ax.set_ylabel("", fontname=fontname, size=fontsize)
    ax.set_ylim([0, math.ceil(maximum)])
    ax.set_title(
        "% of Errors with Confidence ≥ 0.8\n", fontname=fontname, size=fontsize + 2, loc="center", style="italic"
    )

    handles, labels = ax.get_legend_handles_labels()

    plt.xticks(fontname=fontname, fontsize=fontsize)

    yticks = list(range(int(ax.get_ylim()[0]), int(ax.get_ylim()[1]) + 30, 25))
    yticklabels = [f"{y}%" for y in yticks]
    plt.yticks(yticks, yticklabels, fontname=fontname, fontsize=fontsize)
    ax.yaxis.tick_right()

    pad = 0.025 if maximum < 4 else 0.1
    pad = pad if maximum < 50 else 0.5
    for index, p in enumerate(percentages):
        ax.text(index + 0.31, p + pad + 0.5, f"{p:.2f}%", size=fontsize, fontname=fontname)

    ax.set_xticks([x + 0.45 for x in range(0, len(compareOrder))], compareOrder, size=fontsize, fontname=fontname)
    plt.tick_params(bottom=False)
    return fig

class analysis(RAIEVanalysis):
    """ 
    ------------------------------------------
    
    
    Analysis class for the Performance Equity Workflow
    
    Want a recommendation of analysis functions to explore next? Try calling the ___ .suggestWorkflow() ___ method!
    

    ------------------------------------------
    """
    def __init__(
        self,
        predictions,
        *,
        taskType="classification",
        taskCol='predType',
        modelCol="model_alias",
        testSetCol="dataset",
        predID="id",
        confidenceCol="confidence",
        correctCol="correct",
        highConfThreshold=0.9,
        goldCol="gold",
        predCol="pred",
        positiveLabel=None,
        negativeLabel=None,
        predEncodedCol=None,
             log_savedir=None,
             loadLastLog=False,
             loadLogByName=None,
             logger = None,  
        interactive=True
             ):
        """
        Initializing Analysis Object
        """
        
        ## logging prep
        # pull copy of parameters 
        params = locals() 
        params = cleanParams(params, removeFields=['self','predictions'])
        ##
        
        ## logging 
        RAIEVanalysis.__init__(self, log_savedir, load_last_session=loadLastLog, load_session_name=loadLogByName,
                               logger=logger,
                               workflow_outline=WORKFLOW_OUTLINE, interactive=interactive) 
        ##
        
        # log initialization 
        self._logFunc(WORKFLOW_CATEGORY,params=params) 

        self.predictions = predictions
        self.taskType = taskType
        self.modelCol = modelCol
        self.testSetCol = testSetCol
        self.predID = predID
        self.taskCol = taskCol
        self.confidenceCol = confidenceCol
        self.correctCol = correctCol
        self.highConfThreshold = highConfThreshold
        self.goldCol = goldCol
        self.predCol = predCol
        self.CORRECTCOL = "Correct"

        self._annotateConfidenceRanking()
        self.predictions[self.CORRECTCOL] = self.predictions.apply(
            lambda x: "Correct" if x[self.predCol] == x[self.goldCol] else "Incorrect", axis=1
        )
        self.positiveLabel = positiveLabel
        self.negativeLabel = negativeLabel

        # Add numeric column encoding predID; distinct for binary and multiclass cases
        self.predEncodedCol = predEncodedCol
        if self.predEncodedCol is None:
            self.predEncodedCol = "predEncoded"
            classes = list(self.predictions[self.goldCol].unique())
            class_id_dict = {}
            if len(classes) > 2:
                try:
                    classes.remove(self.positiveLabel)
                    classes.remove(self.negativeLabel)
                except ValueError:
                    pass
                class_id_dict[self.positiveLabel] = 1
                class_id_dict[self.negativeLabel] = 0

            class_id_dict.update({n: k for n, k in zip(classes, range(len(classes)))})
            self.predictions[self.predEncodedCol] = self.predictions[self.predCol].map(class_id_dict)

    def workflow(self):
        """ 
        Check Default Workflow Recommendation
        """
        self._logFunc(WORKFLOW_CATEGORY, params=locals()) 
        workflow()
        
    def _annotateConfidenceRanking(self):
        """ 
        Rank confidences and assign to specified colors.
        """

        colors = list(sns.color_palette("tab20c")[4:8]) + list(sns.color_palette("tab20b"))[1:4][::-1]
        color_dict = {n: c for (n, _), c in zip(UNCERTAINTY_RANKINGS, colors)}

        def annotate_confidence(x):
            for rank, confidence_max in UNCERTAINTY_RANKINGS:
                if x < confidence_max:
                    return rank
            return UNCERTAINTY_RANKINGS[-1][0]

        self.predictions["confidence ranking"] = self.predictions[self.confidenceCol].apply(
            lambda x: annotate_confidence(x))
        self.predictions["color"] = self.predictions["confidence ranking"].apply(lambda x: color_dict[x])

    def uncertaintyAcrossCorrectness(self, fontname='sans-serif', fontsize=18):
        """ 
        Analyze distribution of model confidence values across analyst uncertainty bins for correct vs. incorrect predictions.
        
        :param fontname: (str) fontname to use in plotting, default is "sans-serif".
        :param fontsize: (int) fontsize to use in plotting, default is 18.
        """
        self._logFunc(WORKFLOW_CATEGORY, params=locals()) 
        
        for (testSet, model), predDF in self.predictions.groupby([self.testSetCol, self.modelCol]):
            ## display model title
            IPython.display.display(IPython.display.HTML(f"<h4>{model} ({testSet})</h4>"))

            fig, ax = plt.subplots(figsize=(8, 4))
            correct_dict = predDF[self.CORRECTCOL].value_counts().to_dict()
            compareCols = [self.CORRECTCOL]
            cols = compareCols + ["confidence ranking"]
            bar_df = predDF.assign(N=1).groupby(cols, as_index=False)["N"].count()
            bar_df["%"] = bar_df.apply(lambda x: (x["N"] / correct_dict[x["Correct"]]) * 100, axis=1)
            bar_df = bar_df.pivot_table(index=compareCols, columns="confidence ranking", values="%")
            bar_df[[x[0] for x in UNCERTAINTY_RANKINGS if x[0] in bar_df.columns]][::-1].plot.barh(
                stacked=True, ax=ax, color=color_dict, width=0.89
            )

            ax.tick_params(labelrotation=0)
            # ax.set_xlabel('Frequency (%)', fontname=fontname, fontsize=fontsize)
            ax.set_ylabel("")
            # ax.set_title('Uncertainty by Prediction Correctness', fontname=fontname, fontsize=fontsize+2, loc='center', style='italic', )

            handles, labels = ax.get_legend_handles_labels()
            labels = ["roughly\neven chance", "likely", "very likely", "almost certainly"]
            ax.legend(
                handles[::-1],
                labels[::-1],
                title="",
                bbox_to_anchor=[0.98, 0.9],
                prop={"family": fontname, "size": fontsize},
                frameon=False,
            )

            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["left"].set_visible(False)
            xticks = list(range(0, 120, 20))
            xticklabels = [f"{x}%" for x in xticks]
            plt.xticks(xticks, xticklabels, fontname=fontname, fontsize=fontsize)
            plt.yticks(fontname=fontname, fontsize=fontsize)
            # ax.xaxis.set_ticks_position('none')
            ax.yaxis.set_ticks_position("none")
            plt.show()

    def uncertaintyForHighConfidenceErrors(self, fontname="sans-serif", fontsize=18):
        """ 
        Analyze distribution of model confidence values across analyst uncertainty bins for high confidence errors
        
        :param fontname: (str) fontname to use in plotting, default is "sans-serif".
        :param fontsize: (int) fontsize to use in plotting, default is 18.
        """
        self._logFunc(WORKFLOW_CATEGORY, params=locals()) 
        
        if self.positiveLabel is None or self.negativeLabel is None:
            raise Exception(
                "To run this analysis you need to specify positiveLabel and negativeLabel parameters on instantiation"
            )
        for (testSet, model), predDF in self.predictions.groupby([self.testSetCol, self.modelCol]):
            ## display model title
            IPython.display.display(IPython.display.HTML(f"<h4>{model} ({testSet})</h4>"))
            _vertUncertaintyBars(predDF, self.correctCol, self.goldCol, self.predCol, self.positiveLabel,
                                 self.negativeLabel)
            plt.show()

    def applyOutcomeType(self):
        """
        Create 'Outcome Type' column based on goldCol and predCol.
        """
        def _OutcomeType(row):
            if row[self.goldCol] == self.positiveLabel and row[self.predCol] == self.positiveLabel:
                return "True Positive"
            elif row[self.goldCol] == self.negativeLabel and row[self.predCol] == self.positiveLabel:
                return "False Positive"
            elif row[self.goldCol] == self.positiveLabel and row[self.predCol] == self.negativeLabel:
                return "False Negative"
            else:
                return "True Negative"

        self.predictions["Outcome Type"] = self.predictions.apply(_OutcomeType, axis=1)

    def keywordSignificance(self, keywords, task="binary", stat="mw", alternative="greater", confidenceCol=None):
        """
        Calculate statistical significance of keywords.

        :param keywords: (list) Keywords from documents.
        :param task: (str) optional, default "binary". Either "binary" or "multiclass"
        :param stat: (str) optional, default "mw". Name of significance test. One of mw, ks-pdf, ks-cdf, wd-pdf, wd-cdf.
        :param alternative: (str) optional, default "greater". Direction of the alternative hypothesis.
        :param confidenceCol: (str) optional, default None. Name of column for 1 vs Rest testing.
        """
        self._logFunc(WORKFLOW_CATEGORY, params=locals())

        if alternative == "greater":
            alt_char = " > "
        else:
            alt_char = " < "

        stats_results = []
        task_df = self.predictions.loc[self.predictions[self.taskCol] == task]
        for key in keywords:
            if task == "binary":
                if confidenceCol is None:
                    stats_df = stat_table(task_df, compareCol=key, confidenceCol=self.predEncodedCol,
                                          stat=stat, alternative=alternative)
                else:
                    stats_df = []
                    for groupName in task_df[confidenceCol].unique():
                        task_df["1vsRest"] = task_df[confidenceCol].apply(lambda x: 1 if x == groupName else 0)
                        temp = stat_table(task_df, compareCol=key, confidenceCol="1vsRest",
                                          stat=stat, alternative=alternative)
                        temp["PositiveClass"] = groupName
                        stats_df.append(temp)
                    stats_df = pd.concat(stats_df)
            else:
                stats_df = []
                for class_name in task_df[self.predCol].unique():
                    task_df["1vsRest"] = task_df[self.predCol].apply(lambda x: 1 if x == class_name else 0)
                    temp = stat_table(task_df, compareCol=key, confidenceCol="1vsRest",
                                      stat=stat, alternative=alternative)
                    temp["PositiveClass"] = class_name
                    stats_df.append(temp)
                stats_df = pd.concat(stats_df)
            stats_df['keyword'] = key
            stats_results.append(stats_df)
        stats_results = pd.concat(stats_results)
        stats_results = stats_results.reset_index(drop=False)
        stats_results["HA1"] = stats_results["index"].apply(lambda x: x.split(alt_char)[0])
        stats_results["HA2"] = stats_results["index"].apply(lambda x: x.split(alt_char)[1])

        stats_results["HA1"] = stats_results["HA1"].replace({"0.0": "not contains", "1.0": "contains"})
        stats_results["HA2"] = stats_results["HA2"].replace({"0.0": "not contains", "1.0": "contains"})

        return stats_results.set_index("index")

    def visualizeSignificantKeywords(self, results, positive_class, pvalue_threshold=0.05, alt="contains",
                                     remove_prefix=""):
        """
        Create word clouds from significant keywords.

        :param results: (Pandas DataFrame) DataFrame with results from significance testing.
        :param positive_class: (list) List containing the positive class label (binary) or each class to be shown (multi)
        :param pvalue_threshold: (float) optional, default 0.05. Threshold of p-value in sig. testing.
        :param alt: (str) optional, default "contains". Either "contains" or "not contains"
        :param remove_prefix: (str) optional, default "". If keywords have undesirable prefixes.
        """
        self._logFunc(WORKFLOW_CATEGORY, params=locals(), removeParamFields=['results'])
        results['keyword'] = results['keyword'].apply(
            lambda x: x[len(remove_prefix):] if x.startswith(remove_prefix) else x)
        results['keyword'] = results['keyword'].apply(lambda x: "-".join(x.split(" ")))
        if len(positive_class) == 1:  # Binary task
            words = results.loc[(results['pvalue'] < pvalue_threshold) & (results['HA1'] == alt)]
            text = " ".join(i for i in words['keyword'])
            wordcloud = WordCloud(background_color="white", regexp=r"\S*").generate(text)
            plt.figure(figsize=(15, 10))
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis("off")
            plt.title(
                f"When a document {alt} these keywords, the document is predicted as \n{positive_class[0]} significantly MORE than documents that don't {alt} these keywords.")
        else:  # Multiclass task
            if (len(positive_class) % 2 == 0):
                fig, axes = plt.subplots(len(positive_class) // 2, 2, figsize=(20, 10))
                axes = axes.flatten()
            else:
                fig, axes = plt.subplots(len(positive_class) // 3, 3, figsize=(20, 10))
                axes = axes.flatten()
            # Generate word cloud for each class
            for j, i in enumerate(positive_class):
                grp = results.loc[results["PositiveClass"] == i]
                et_words = grp.loc[(grp['pvalue'] < pvalue_threshold) & (grp['HA1'] == alt)]
                text = " ".join(w for w in et_words['keyword'])
                wordcloud = WordCloud(background_color="white", regexp=r"\S*").generate(text)
                axes[j].imshow(wordcloud, interpolation='bilinear')
                axes[j].axis("off")
                axes[j].set_title(i, y=-0.1)
            plt.suptitle(
                f"When a document {alt} these keywords, the document is predicted as \nSUBCLASS significantly MORE than documents that don't {alt} these keywords.")
        plt.show()
