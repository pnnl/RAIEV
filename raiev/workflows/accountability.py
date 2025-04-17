import IPython
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from ..vis.acc import confusion_matrix_comparisons
from ..vis.basics import barplot_grid

from .base import RAIEVanalysis, cleanParams   
import inspect

from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.express as px

WORKFLOW_CATEGORY = "Accountability Overview"
WORKFLOW_DESC = "Accountability Overview evaluations capture aggregate performance (leveraged in traditional evaluations) and targeted evaluations of model performance across critical subsets or circumstances."

WORKFLOW_OUTLINE = """## import the raiev package
import raiev
## import the accountability workflow sub-module
from raiev.workflows import accountability

## Load model predictions
predictions = load_predictions(filepath_to_predictions)
    
## Instantiation of workflow object
acc = accountability.analysis(predictions, taskType = 'classification',
                              taskCol='predType', modelCol = 'model_alias',
                              testSetCol = 'dataset', goldCol = 'gold', predCol = 'pred',
                              predConfidence = 'confidence', predID = 'id', display=True)

## Summary table of aggregate metrics across models and test sets
acc.agg_metrics()

## Aggregate metrics contrasting model performance within test set(s) using bar plots
acc.plot_agg_metrics()

## Confusion matrices highlighting error types (e.g., misclassifications between classes)
acc.confusion_matrix()

## Interactive Confusion matrices highlighting error types (e.g., misclassifications between classes)

## Continued Analysis, Analytic Interface(s), etc. under development
"""


def workflow():
    """
    Display example workflow code for Accountability Overviews
    """
    IPython.display.display(IPython.display.HTML(f"<h3>Overview of {WORKFLOW_CATEGORY} Workflow</h3>"))
    print(WORKFLOW_DESC)
    IPython.display.display(IPython.display.HTML("<b>Implemented workflow includes:</b>"))
    print(WORKFLOW_OUTLINE)


class analysis(RAIEVanalysis):
    """ 
    ------------------------------------------
    
    
    Analysis class for Accountability Overviews
    
    Want a recommendation of analysis functions to explore next? Try calling the ___ .suggestWorkflow() ___ method!
    

    ------------------------------------------
    """

    def __init__(
            self,
            predictions,
            *,
            taskType="classification",
            taskCol="predType",
            modelCol="model_alias",
            testSetCol="dataset",
            goldCol="gold",
            predCol="pred",
            predConfidence="confidence",
            predID="id",
            display=False,
            round2=3,
            #
            predictedRankCol="pred_rank",
            sourceTypeCol="Source Type",
            targetTypeCol="Target Type",
            log_savedir=None,
            loadLastLog=False,
            loadLogByName=None,
            logger=None,
            interactive=True
    ):
        """
        Initializing Analysis Object
        """

        ## logging prep
        # pull copy of parameters 
        params = locals()
        params = cleanParams(params, removeFields=['self', 'predictions'])
        ##

        ## logging 
        RAIEVanalysis.__init__(self, log_savedir, load_last_session=loadLastLog, load_session_name=loadLogByName,
                               logger=logger,
                               workflow_outline=WORKFLOW_OUTLINE, interactive=interactive)
        ##
        self._logFunc(WORKFLOW_CATEGORY, params=params)

        self.predictions = predictions
        self.taskType = taskType
        self.taskCol = taskCol
        self.modelCol = modelCol
        self.testSetCol = testSetCol

        if type(self.testSetCol) is list:
            def test_set_rec(x, testSetCols):
                return "-".join([x[i] for i in testSetCols])

            self.predictions["TEST_SET"] = self.predictions.apply(lambda x: test_set_rec(x, self.testSetCol), axis=1)
            self.testSetCol = "TEST_SET"

        self.goldCol = goldCol
        self.predCol = predCol
        self.predConfidence = predConfidence

        self.predictedRankCol = predictedRankCol
        self.sourceTypeCol = sourceTypeCol
        self.targetTypeCol = targetTypeCol
        self.LINKTYPECOL = "Link Type"

        self.predID = predID
        self.display = display
        self.round2 = round2
        self._agg_metrics = None
        self._agg_metrics_cols = None

    def workflow(self):
        """ 
        Check Default Workflow Recommendation
        """
        self._logFunc(WORKFLOW_CATEGORY, params=locals())
        workflow()

    def agg_metrics(self, round2=None, displayTable=True, sort_col="F1"):
        """
        Summarize Aggregate Metrics Results

        :param round2: (int) number of significant figures to round results to
        :param displayTable: (boolean) indicating whether to display the table of results inline or not, default is True
        :param sort_col: (str) optional, default F1. Specifies which metric to sort the table by.
        """
        ## log  
        self._logFunc(WORKFLOW_CATEGORY, params=locals()) 

        if round2 is None:
            round2 = self.round2
        if displayTable is None:
            displayTable = self.display
        if self.taskType == "classification":
            self._agg_metrics_cols = []
            results = []
            for (testSet, model), predDF in self.predictions.groupby([self.testSetCol, self.modelCol]):
                #
                predictedLabels = predDF[self.predCol].tolist()
                goldLabels = predDF[self.goldCol].tolist()
                p, r, f, _ = precision_recall_fscore_support(
                    predictedLabels, goldLabels, average="macro", labels=np.unique(predictedLabels), zero_division=0
                )
                results.append(
                    {
                        self.testSetCol: testSet,   
                        self.modelCol: model,   
                        "Precision": round(p, round2),
                        "Recall": round(r, round2),
                        "F1": round(f, round2),
                    }
                )
            results = pd.DataFrame(results)
            if displayTable:
                IPython.display.display(IPython.display.HTML("<h3>Aggregate Metrics Summary Table</h3>"))
                IPython.display.display(
                    results.rename(columns={self.testSetCol: "Test Set", self.modelCol: "Model"}).sort_values(
                        by=sort_col, ascending=False))
            self._agg_metrics = results

        elif self.taskType == "link-prediction":
            LINKTYPECOL = self.LINKTYPECOL
            self.predictions[LINKTYPECOL] = (
                    self.predictions[self.sourceTypeCol] + " to " + self.predictions[self.targetTypeCol]
            )
            self.predictions["recip_rank"] = 1 / self.predictions[self.predictedRankCol]

            def rankMetricRes(predictions, modelCol, LINKTYPECOL, col2avg, metricLbl):
                metrics = pd.concat(
                    [
                        predictions.groupby(modelCol)[col2avg]
                        .mean()
                        .reset_index()
                        .rename(columns={col2avg: metricLbl})
                        .assign(edgetype="Overall")
                        .rename(columns={"edgetype": LINKTYPECOL}),
                        predictions.groupby([modelCol, LINKTYPECOL])[col2avg]
                        .mean()
                        .reset_index()
                        .rename(columns={col2avg: metricLbl}),
                    ]
                )
                return metrics

            results = []
            for (testSet, model), predDF in self.predictions.groupby([self.testSetCol, self.modelCol]):
                # mean recipricol rank
                metrics = rankMetricRes(predDF, self.modelCol, LINKTYPECOL, "recip_rank", "MRR")
                metrics = metrics[[self.modelCol, LINKTYPECOL, "MRR"]]
                # mean rank
                # metrics = pd.merge(metrics,
                #                   rankMetricRes(predDF, self.modelCol, LINKTYPECOL, self.predictedRankCol, 'MR'))
                # hits @ k metrics
                for k in [1, 3, 10]:
                    hitsCol = f"Hits @ {k}"
                    predDF[hitsCol] = predDF[self.predictedRankCol].apply(lambda x: int(x <= k))
                    metrics = pd.merge(metrics, rankMetricRes(predDF, self.modelCol, LINKTYPECOL, hitsCol, hitsCol))
                metrics[self.testSetCol] = testSet
                results.append(metrics)
                del metrics
            self._agg_metrics = pd.concat(results)
            if displayTable:
                IPython.display.display(IPython.display.HTML("<h3>Aggregate Metrics Summary Table</h3>"))
                IPython.display.display(self._agg_metrics)
            del results

    def plot_agg_metrics(self, cmap="plasma", use_sns=False):
        """
        Plot Aggregate Metrics Results

        :param cmap: (str) default plasma. Specifies the colormap to use for plotting.
        :param use_sns: (boolean) default False. Indicates whether or not to use seaborn for plotting.
        """
        self._logFunc(WORKFLOW_CATEGORY, params=locals())

        if self._agg_metrics is None:
            self.agg_metrics(displayTable=False)

        ## display header
        IPython.display.display(IPython.display.HTML("<h3>Aggregate Metrics Plot</h3>"))

        hueBy = None
        if self.LINKTYPECOL in self._agg_metrics.columns:
            hueBy = self.LINKTYPECOL
            long_mets = (
                self._agg_metrics.melt(id_vars=[self.testSetCol, self.modelCol, self.LINKTYPECOL])
                .rename(columns={"variable": "metric", self.testSetCol: "Test Set", self.modelCol: "Model"})
                .reset_index()
            )
        else:
            long_mets = self._agg_metrics.melt(id_vars=[self.testSetCol, self.modelCol]).rename(
                columns={"variable": "metric", self.testSetCol: "Test Set", self.modelCol: "Model"}
            )

            long_mets = (
                self._agg_metrics.melt(id_vars=[self.testSetCol, self.modelCol])
                .rename(columns={"variable": "metric", self.testSetCol: "Test Set", self.modelCol: "Model"})
                .reset_index()
            )

        if use_sns:
            if long_mets["Test Set"].nunique() > 1:
                if hueBy is not None:
                    g = sns.FacetGrid(long_mets, row="metric", col="Test Set", hue=hueBy)
                else:
                    g = sns.FacetGrid(long_mets, row="metric", col="Test Set")
            else:
                if hueBy is not None:
                    g = sns.FacetGrid(long_mets, row="metric", col=hueBy)

                else:
                    g = sns.FacetGrid(long_mets, col="metric")

            g.map(sns.barplot, "Model", "value", order=sorted(long_mets["Model"].unique()))
            g.set_titles(col_template="{col_name}", row_template="{row_name}")
            g.set_xticklabels(rotation=30)

            plt.ylim(0, 1)
            plt.show()
        else:
            # plotly barplot grid
            barplot_grid(long_mets, "Model", "value", "Test Set", "metric", cmap=cmap, ylims=[0, 1])

    def _confusion_matrix(self, frame, subset=None, cmap="Reds"):
        def confmatrix(frame):
            gb = [self.goldCol, self.predCol]
            hm = frame.groupby(gb)[[self.predID]].nunique().reset_index().sort_values(by=gb)
            hm = hm.pivot_table(index=self.goldCol, columns=self.predCol, values=self.predID)
            return hm

        colormap = sns.color_palette(cmap)

        if subset is None:
            df_cm = confmatrix(frame)
            labels = list(df_cm.columns)
            if len(labels) > 10:
                dim = len(labels) / 4
                fig, ax = plt.subplots(1, figsize=(dim, dim))
                sns.heatmap(df_cm, ax=ax, cmap=colormap)
            else:
                fig, ax = plt.subplots(1)
                sns.heatmap(df_cm, annot=True, fmt="g", ax=ax, cmap=colormap)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
        else:
            # if subset is not None:
            if isinstance(subset, list):
                num_plots = np.prod([frame[n].nunique() for n in subset])
            else:
                num_plots = frame[subset].nunique()
            fig, axes = plt.subplots(num_plots, figsize=(8, 8))
            axes = axes.flatten()
            for i, (j, grp) in enumerate(frame.groupby(subset)):
                df_cm = confmatrix(grp)
                sns.heatmap(df_cm, annot=True, fmt="g", ax=axes[i], cmap=colormap)
                axes[i].set_xlabel("Predicted")
                axes[i].set_ylabel("Actual")
                axes[i].set_title(f"{subset} = {j}")
                fig.tight_layout()

    def confusion_matrix(self, subset=None, cmap="Reds"):
        """
        Plot a static confusion matrix. Each row represents the instances in an actual class, distributed across columns to reflect what the model predicted the instances to be. This visual highlights how a model confuses classes (i.e., mislabels instances of one class as another class)

        :param subset: (str) optional, default None. Column in predictions to create subsets of confusion matrices.
        :param cmap: (str) colormap to use in plot
        """
        self._logFunc(WORKFLOW_CATEGORY, params=locals())  # , removeParamFields=[])

        if self.taskType == "classification":
            ## display confusion matrix header
            IPython.display.display(IPython.display.HTML("<h3>Comparison Confusion Matrix</h3>"))
            desc = """Each row represents the instances in an actual class, distributed across columns to reflect what the model predicted the instances to be. This visual highlights how a model confuses classes (i.e., mislabels instances of one class as another class)"""
            IPython.display.display(IPython.display.HTML(f"<p>{desc}</p>"))

            results = []
            for (testSet, model), predDF in self.predictions.groupby([self.testSetCol, self.modelCol]):
                ## display confusion matrix sub-title
                IPython.display.display(IPython.display.HTML(f"<h4>{model} ({testSet})</h4>"))
                ## display confusion matrix
                self._confusion_matrix(predDF, subset=subset, cmap=cmap)
                plt.title(f"{model} ({testSet})")
                plt.show()
        else:
            print("Confusion matrices are only applicable for classification tasks")

    def confusion_matrix_comparison(self, cmap="plasma", comparisonCol=""):
        """
        Generate an interactive confusion matrix. Each row represents the instances in an actual class, distributed across columns to reflect what the model predicted the instances to be. Barplots inside each cell illustrates the distribution across a third variable. This interactive visual highlights how a model confuses classes (i.e., mislabels instances of one class as another class) and how those outcomes are distributed across the third, comparison variable.

        :param cmap: (str) colormap to use in plot
        :param comparisonCol: (str) column containing the factor to compare across within the matrix
        """
        self._logFunc(WORKFLOW_CATEGORY, params=locals())

        if comparisonCol == "":
            comparisonCol = self.modelCol
        if self.taskType == "classification":
            ## display confusion matrix header
            IPython.display.display(IPython.display.HTML("<h3>Confusion Matrix</h3>"))
            desc = """Each row represents the instances in an actual class, distributed across columns to reflect what the model predicted the instances to be. Barplots inside each cell illustrates the distribution across a third variable. This interactive visual highlights how a model confuses classes (i.e., mislabels instances of one class as another class) and how those outcomes are distributed across the third, comparison variable."""
            IPython.display.display(IPython.display.HTML(f"<p>{desc}</p>"))

            confusion_matrix_comparisons(self.predictions, comparisonCol,
                                         includeOverall=False, onlyOverall=False,
                                         overallColor='skyblue', cmap=cmap,
                                         model_col=self.modelCol,
                                         testset_col=self.testSetCol)
        else:
            print("Confusion matrices are only applicable for classification tasks")
            return
