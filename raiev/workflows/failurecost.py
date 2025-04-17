import IPython
import pandas as pd
import matplotlib.pyplot as plt

from .base import RAIEVanalysis, cleanParams
from raiev.utils import color_prep
from raiev.assistance import clusterModelsByData, projection, characterizeClusters
from raiev.vis import plot_clusters, plot_error_bars
from raiev.vis import add_high_confidence, plot_kde, compare_across, stat_plot, stat_data, error_data

WORKFLOW_CATEGORY = "Failure Cost Characterization"
WORKFLOW_DESC = "Failure Cost Characterization evaluations distinguish errors based on failure costs."

WORKFLOW_OUTLINE = """## import the raiev package
import raiev
## import the Failure Cost Characterization workflow sub-module
from raiev.workflows import failurecost

## Load model predictions
predictions = load_predictions(filepath_to_predictions)
    
## Instantiation of workflow object
fcc = failurecost.analysis(predictions)

## Overview of high confidence errors comparing model performance
fcc.highConfidenceErrors()

## Comparison of high confidence errors across classes for each model
fcc.highConfidenceErrorsByClass()

## Overview of model confidence distributions overall
fcc.confidenceDistributions()

## Overview of model confidence distributions across classes
fcc.classConfidence()

## Highlights of significantly different confidence distributions to examine 
fcc.confidenceComparisons()


## Continued Analysis, Analytic Interface(s), etc. under development

"""


def workflow():
    """"""
    IPython.display.display(IPython.display.HTML(f"<h3>Overview of {WORKFLOW_CATEGORY} Workflow</h3>"))
    print(WORKFLOW_DESC)
    IPython.display.display(IPython.display.HTML("<b>Implemented workflow includes:</b>"))
    print(WORKFLOW_OUTLINE)


class analysis(RAIEVanalysis):
    """ 
    ------------------------------------------
    
    
    Analysis class for the Failure Cost Characterization Workflow
    
    Want a recommendation of analysis functions to explore next? Try calling the ___ .suggestWorkflow() ___ method!
    

    ------------------------------------------
    """

    def __init__(
            self,
            predictions,
            *,
            taskType="classification",
            modelCol="model_alias",
            testSetCol="dataset",
            predID="id",
            confidenceCol="confidence",
            highConfThreshold=0.9,
            goldCol="gold",
            predCol="pred",
            correctCol="correct",
            plotsGroupbyCol=None,
            random_seed=100,
            log_savedir=None,
            loadLastLog=False,
            loadLogByName=None,
            logger=None,
            interactive=True,
            cmap='viridis',
            colors=None
    ):
        """
        Initializing Analysis Object
        """

        # logging prep
        # pull copy of parameters 
        params = locals()
        params = cleanParams(params, removeFields=['self', 'predictions'])
        ##

        # logging
        RAIEVanalysis.__init__(self, log_savedir, load_last_session=loadLastLog, load_session_name=loadLogByName,
                               logger=logger,
                               workflow_outline=WORKFLOW_OUTLINE, interactive=interactive)

        # log initialization 
        self._logFunc(WORKFLOW_CATEGORY, params=params)

        self.predictions = add_high_confidence(predictions, threshold=highConfThreshold, confidenceCol=confidenceCol)
        self.taskType = taskType
        self.modelCol = modelCol
        self.testSetCol = testSetCol
        self.predID = predID
        self.confidenceCol = confidenceCol
        self.correctCol = correctCol
        self.highConfThreshold = highConfThreshold
        self.goldCol = goldCol
        self.predCol = predCol
        self.plotsGroupbyCol = plotsGroupbyCol
        self.random_seed = random_seed
        self.cmap = cmap
        self.colors = colors
        self.clusterDescriptions = {}

        # if plotsGroupbyCol is not unique for each model, make it unique
        if self.plotsGroupbyCol is not None:
            numModelCol = len(set(self.predictions.groupby(self.plotsGroupbyCol,
                                                           as_index=False)[self.modelCol].size()["size"].unique()))
            if numModelCol != 1:
                newCol = f"{self.plotsGroupbyCol}__{self.modelCol}"
                self.predictions[newCol] = self.predictions[self.plotsGroupbyCol] + "__" + self.predictions[
                    self.modelCol]
                self.groupModelCol = newCol
            else:
                self.groupModelCol = self.plotsGroupbyCol
            pltCol = self.plotsGroupbyCol

        else:
            pltCol = self.modelCol

        self.modelColors, self.modelCompareOrder = color_prep.prep_colors_for_bars(self.predictions, self.modelCol,
                                                                                   self.cmap, self.colors)
        self.classColors, self.classCompareOrder = color_prep.prep_colors_for_bars(self.predictions, self.goldCol,
                                                                                   None, None)
        self.colors, self.compareOrder = color_prep.prep_colors_for_bars(self.predictions, pltCol,
                                                                         self.cmap, self.colors)

    def workflow(self):
        """ 
        Check Default Workflow Recommendation
        """
        self._logFunc(WORKFLOW_CATEGORY, params=locals())
        workflow()

    def confidenceDistributions(self, title=None, plotGroups=False, returnGroups_dict=False, returnGroups_df=False,
                                threshold=0.45):
        """ 
        Calculate and display cumulative and probability density distributions of predicted confidence.

        :param title: str.
        :param plotGroups: Boolean (default False).
        :param returnGroups_dict: Boolean (default False).
        :param returnGroups_df: Boolean (default False).
        :param threshold: float (default 0.45).
        """
        self._logFunc(WORKFLOW_CATEGORY, params=locals())

        groupedOrder, groupedUmap = self.groupModelsByPerformance(plot_type="total", modelCol=self.modelCol,
                                                                  plotGroups=plotGroups,
                                                                  threshold=threshold)
        self.predictions["GroupForPlotting"] = self.predictions[self.modelCol].map(groupedOrder)
        groups = set(groupedOrder.values())
        numGroups = len(groups)
        for groupName in groups:
            title = "Model Confidence Distribution" if title is None else title
            if numGroups > 1:
                title += f' - Group {groupName}'

            curr = stat_data(
                self.predictions.loc[self.predictions["GroupForPlotting"] == groupName],
                compareCol=self.modelCol, confidenceCol=self.confidenceCol, round_confidence=2
            )
            stat_plot(curr, dist="cdf", title=title, confidenceCol=self.confidenceCol, compareColor=self.modelColors,
                      compareOrder=self.modelCompareOrder)
            stat_plot(curr, dist="pdf", title=title, confidenceCol=self.confidenceCol, compareColor=self.modelColors,
                      compareOrder=self.modelCompareOrder)

        returnObj = []
        if returnGroups_dict:
            returnObj.append(groupedOrder)
        if returnGroups_df:
            returnObj.append(groupedUmap)
        if len(returnObj) == 1:
            return returnObj[0]
        elif len(returnObj) > 1:
            return returnObj

    def classConfidence(self, plotGroups=False, returnGroups_dict=False, returnGroups_df=False, threshold=0.45):
        """ 
        Calculate and display KDE distributions of predicted confidence per class.
        :param plotGroups: Boolean (default False).
        :param returnGroups_dict: Boolean (default False).
        :param returnGroups_df: Boolean (default False).
        :param threshold: float (default 0.45).
        """
        self._logFunc(WORKFLOW_CATEGORY, params=locals())

        # self.LOG.info('classConfidence')
        groupedOrder, groupedUmap = self.groupModelsByPerformance(plot_type="total", modelCol=self.modelCol,
                                                                  barBinCol=self.modelCol,
                                                                  plotGroups=plotGroups,
                                                                  threshold=threshold)
        self.predictions["GroupForPlotting"] = self.predictions[self.modelCol].map(groupedOrder)
        groups = set(groupedOrder.values())
        numGroups = len(groups)
        try:
            self.predictions["Gold"] = self.predictions[self.goldCol].apply(lambda x: f"Gold = {x.title()}")
        except AttributeError:
            self.predictions["Gold"] = self.predictions[self.goldCol].apply(lambda x: f"Gold = {x}")

        for groupName in groups:
            subset = self.predictions.loc[self.predictions["GroupForPlotting"] == groupName]
            for (test, model), mdf in subset.groupby([self.testSetCol, self.modelCol]):
                title = f"{model} on {test}\nConfidence Distribution"
                if numGroups > 1:
                    new_title = title + " - Group " + str(groupName)
                else:
                    new_title = title
                # Generate KDE Line plot for subsets of same model data.
                compare_across(
                    mdf,
                    confidenceCol=self.confidenceCol,
                    compareCol="Gold",
                    compareOrder=None,
                    overall=True,
                    title=new_title,
                    colors=["black", "tab:blue", "tab:orange"],
                    lines=None,
                    fig_kwargs={},
                )
        returnObj = []
        if returnGroups_dict:
            returnObj.append(groupedOrder)
        if returnGroups_df:
            returnObj.append(groupedUmap)
        if len(returnObj) == 1:
            return returnObj[0]
        elif len(returnObj) > 1:
            return returnObj

    def groupModelsByPerformance(self, plot_type, modelCol, df=None, barBinCol=None, annot=True, overall=False,
                                 plotGroups=False, threshold=0.45, noCluster=False):
        """Calculate how models should be grouped for nice plotting.
        1) Groups created using self.plotsGroupbyCol (e.g. treatment)
        2) Automatically group models using kmeans (always done if more than 5 models and self.plotsGroupbyCol=None)
        3) No grouping; plot as is.

        :param plot_type: str. error or total.
        :param modelCol: str.
        :param df: DataFrame.
        :param barBinCol: str (optional, default None). Name of column for binning.
        :param annot: Boolean (optional, default True).
        :param overall: Boolean (optional, default False). Indicates if overall error bar should be shown (per class).
        :param plotGroups: Boolean (optional, default False).
                    Indicates if models should be grouped using plotsGroupbyCol.
        :param threshold: float (optional, default 0.45). Value to determine number of groups in kmeans.
        :param noCluster: Boolean (optional, default False). Turn off automatic clustering.
        """
        self._logFunc(WORKFLOW_CATEGORY, params=locals())

        plot_type = "error" if plot_type not in ["error", "total"] else plot_type
        annot = annot if plot_type == "error" else False
        overall = overall if plot_type == "error" else False

        if plotGroups:
            if self.plotsGroupbyCol is None:
                raise ValueError("Must specify plotsGroupbyCol in workflow object to plot groups or "
                                 "set plotGroups to False.")
            if barBinCol is None:
                barBinCol = self.groupModelCol
        else:
            if barBinCol is None:
                barBinCol = modelCol
        # barBinCol = barBinCol if barBinCol is not None else modelCol
        if df is None:
            df = self.predictions.copy()

        if overall:
            data = []
            for (test, model), mdf in df.groupby([self.testSetCol, modelCol]):
                compareOrder = sorted(list(mdf[barBinCol].unique()))
                output, _ = error_data(mdf, barBinCol, plot_type=plot_type, correctCol=self.correctCol,
                                       highConfidenceCol='high confidence', compareOrder=compareOrder, perModel=False,
                                       overall=True, sort=False)
                output[modelCol] = model
                output[self.testSetCol] = test
                data.append(output)
            data = pd.concat(data)
        else:
            compareOrder = sorted(list(df[barBinCol].unique()))
            data, _ = error_data(df, barBinCol, plot_type=plot_type, correctCol=self.correctCol,
                                 highConfidenceCol='high confidence', compareOrder=compareOrder, perModel=True,
                                 overall=False, sort=False)
        if plotGroups:
            # Add column(s) back
            if isinstance(self.plotsGroupbyCol, list):
                for col in self.plotsGroupbyCol:
                    data[col] = data[barBinCol].map(
                        df.set_index(barBinCol)[col].to_dict())
            else:
                data[modelCol] = data[barBinCol].map(
                    df.set_index(barBinCol)[modelCol].to_dict())
        if plotGroups:
            if isinstance(self.plotsGroupbyCol, list):
                data["tempCol"] = data[self.plotsGroupbyCol].apply(lambda x: ' '.join(x), axis=1)
                groupOrder = data.set_index(modelCol)["tempCol"].to_dict()
                data = data.drop(columns=["tempCol"])
            else:
                groupOrder = data.set_index(self.groupModelCol)[modelCol].to_dict()
        # Turn off automatic clustering
        elif noCluster:
            groupOrder = {k: 0 for k in data[modelCol].tolist()}
        # Automatically perform cluster on data with more than 5 models
        elif data[barBinCol].nunique() > 5:
            data = clusterModelsByData(data, modelCol, self.random_seed, threshold)
            IPython.display.display(
                IPython.display.HTML(
                    "<h5>AI Assisted Clustering</h5>"))
            groupOrder = data.set_index(modelCol)["Ranked Group"].to_dict()
        else:
            groupOrder = {k: 0 for k in data[modelCol].tolist()}

        return groupOrder, data.set_index(barBinCol)

    def highConfidenceErrors(self, plotGroups=False, ticklabels=True, sort=True, horizontal=True,
                             returnGroups_dict=False, returnGroups_df=False, threshold=0.45, noCluster=False,
                             verbose=True):
        """ 
        Calculate and display high confident error rates.
        :param plotGroups: Boolean (default False). Indicates if models should be grouped using plotsGroupbyCol.
        :param sort: Boolean (default True). Sorts the bar plot.
        :param horizontal: Boolean (default True). Horizontal or vertical bar plot.
        :param ticklabels: Boolean (default True). To display tick labels
        :param returnGroups_dict: Boolean (default False). To return dictionary of model to group mappings.
        :param returnGroups_df: Boolean (default False). To return dataframe of groupings.
        :param threshold: float (default 0.45). Value of clustering cutoff for kmeans.
        :param noCluster: Boolean (optional, default False). Turn off automatic clustering.
        :param verbose: Boolean (optional, default True). Prints natural language description of clusters.
        """
        self._logFunc(WORKFLOW_CATEGORY, params=locals())

        groupedOrder, groupedUmap = self.groupModelsByPerformance(plot_type="total", modelCol=self.modelCol,
                                                                  plotGroups=plotGroups,
                                                                  threshold=threshold, noCluster=noCluster)
        if plotGroups:
            self.predictions["GroupForPlotting"] = self.predictions[self.groupModelCol].map(groupedOrder)
            pltCol = self.groupModelCol
            colors = self.colors
            compareOrder = self.compareOrder
        else:
            self.predictions["GroupForPlotting"] = self.predictions[self.modelCol].map(groupedOrder)
            pltCol = self.modelCol
            colors = self.modelColors
            compareOrder = self.modelCompareOrder
        groups = set(groupedOrder.values())
        numGroups = len(groups)

        perModel = True if numGroups > 1 else False
        cDescKeys = self.clusterDescriptions.keys()
        for groupName in groups:
            if groupName not in cDescKeys:
                self.clusterDescriptions[f"Group {groupName}"] = {}
            df = self.predictions[self.predictions["GroupForPlotting"] == groupName].reset_index(drop=True)
            if plotGroups and self.groupModelCol != self.plotsGroupbyCol:
                df[self.groupModelCol] = df[self.groupModelCol].apply(lambda x: x.split("__")[0])

            error_title = "High Confidence Errors Relative to Total Errors"
            total_title = "High Confidence Errors Relative to Total Predictions"
            if numGroups > 1:
                error_title += f" - Group {groupName}"
                total_title += f" - Group {groupName}"

            IPython.display.display(
                IPython.display.HTML(
                    "<h4>Illustrating the error rate, highlighting in bold the high confidence errors:</h4>"))

            data, compareOrder_updated = error_data(df, pltCol, plot_type="total", correctCol=self.correctCol,
                                                    highConfidenceCol="high confidence", perModel=perModel,
                                                    compareOrder=compareOrder, colors=colors, overall=False, sort=True)
            # Get descriptions if previously called
            try:
                desc_a = self.clusterDescriptions[f"Group {groupName}"]
            except KeyError:
                desc_a = None

            # Generate Bar plot showing distribution of high confidence errors over total predictions.
            cluster_description_a = plot_error_bars(
                data,
                pltCol,
                compareOrder=compareOrder_updated,
                plot_type="total",
                title=total_title,
                ticklabels=ticklabels,
                horizontal=horizontal,
                descName="highConfidenceErrors",
                desc=desc_a,
                verbose=verbose)

            IPython.display.display(
                IPython.display.HTML(
                    "<h4>Comparing the relative percentage of errors that are high confidence across models:</h4>"
                )
            )
            data, compareOrder_updated = error_data(df, pltCol, plot_type="error", correctCol=self.correctCol,
                                                    highConfidenceCol="high confidence", perModel=perModel,
                                                    compareOrder=compareOrder, colors=colors, overall=False, sort=True)

            # Generate Bar plot showing high confidence errors over total errors.
            cluster_description_b = plot_error_bars(
                data,
                pltCol,
                compareOrder=compareOrder_updated,
                plot_type="error",
                title=error_title,
                ticklabels=ticklabels,
                horizontal=horizontal,
                descName="highConfidenceErrorsOnly",
                desc=desc_a,
                verbose=verbose)

            # Save descriptions if function called again
            if len(desc_a.keys()) == 0:
                self.clusterDescriptions[f"Group {groupName}"]["highConfidenceErrors"] = cluster_description_a
                self.clusterDescriptions[f"Group {groupName}"]["highConfidenceErrorsOnly"] = cluster_description_b

        returnObj = []
        if returnGroups_dict:
            returnObj.append(groupedOrder)
        if returnGroups_df:
            returnObj.append(groupedUmap)
        if len(returnObj) == 1:
            return returnObj[0]
        elif len(returnObj) > 1:
            return returnObj

    def highConfidenceErrorsByClass(self, barBinCol=None,
                                    plotGroups=False,
                                    ticklabels=True, returnGroups_dict=False,
                                    returnGroups_df=False, threshold=0.45, verbose=True):
        """ 
        Calculate and display high confident error rates by class.
        :param barBinCol: str (default self.goldCol). Name of column with true class labels.
        :param plotGroups: Boolean (default False). Indicates if models should be grouped using plotsGroupbyCol.
        :param ticklabels: Boolean (default True). To display tick labels
        :param returnGroups_dict: Boolean (default False). To return dictionary of model to group mappings.
        :param returnGroups_df: Boolean (default False). To return dataframe of groupings.
        :param threshold: float (default 0.45). Value of clustering cutoff for kmeans.
        :param verbose: Boolean (optional, default True). Prints natural language description of clusters.
        """
        self._logFunc(WORKFLOW_CATEGORY, params=locals())

        barBinCol = barBinCol if barBinCol is not None else self.goldCol
        groupedOrder, groupedUmap = self.groupModelsByPerformance(plot_type="error", modelCol=self.modelCol,
                                                                  barBinCol=barBinCol,
                                                                  plotGroups=plotGroups,
                                                                  overall=True, threshold=threshold)
        self.predictions["GroupForPlotting"] = self.predictions[self.modelCol].map(groupedOrder)
        groups = set(groupedOrder.values())
        numGroups = len(groups)
        cDescKeys = self.clusterDescriptions.keys()
        for groupName in groups:
            if groupName not in cDescKeys:
                self.clusterDescriptions[f"Group {groupName}"] = {"highConfidenceErrorsByClass": {}}
            elif "highConfidenceErrorsByClass" not in self.clusterDescriptions[f"Group {groupName}"].keys():
                self.clusterDescriptions[f"Group {groupName}"]["highConfidenceErrorsByClass"] = {}
            subset = self.predictions.loc[self.predictions["GroupForPlotting"] == groupName]
            IPython.display.display(
                IPython.display.HTML(
                    "<h4>Comparing the relative percentage of errors that are high confidence across classes for each model</h4>"
                )
            )
            keys = self.clusterDescriptions[f"Group {groupName}"].keys()
            for (test, model), mdf in subset.groupby([self.testSetCol, self.modelCol]):
                if f"{test}, {model}" not in keys:
                    self.clusterDescriptions[f"Group {groupName}"]["highConfidenceErrorsByClass"][f"{test}, {model}"] = {}
                error_title = "High Confidence Errors Relative to Total Errors"
                total_title = "High Confidence Errors Relative to Total Predictions"

                title = error_title

                if numGroups > 1:
                    title += f" - Group {groupName}"
 
                data, compareOrder_updated = error_data(mdf, barBinCol, plot_type="error", correctCol=self.correctCol,
                                                        highConfidenceCol="high confidence", perModel=False,
                                                        compareOrder=self.classCompareOrder, colors=self.classColors,
                                                        overall=True,
                                                        sort=False)

                try:
                    desc_a = self.clusterDescriptions[f"Group {groupName}"]["highConfidenceErrorsByClass"][f"{test}, {model}"]
                except KeyError:
                    desc_a = None

                # Generate Bar plot showing high confidence errors over total errors.
                cluster_description = plot_error_bars(
                    data,
                    barBinCol,
                    plot_type="error",
                    compareOrder=compareOrder_updated,
                    title=title,
                    ticklabels=ticklabels,
                    annot=True,
                    log=False,
                    horizontal=True,
                    descName="highConfidenceErrorsByClass",
                    desc=desc_a,
                    verbose=verbose)

                # Save descriptions if function called again
                if len(desc_a.keys()) == 0:
                    self.clusterDescriptions[f"Group {groupName}"]["highConfidenceErrorsByClass"][f"{test}, {model}"] = cluster_description

        returnObj = []
        if returnGroups_dict:
            returnObj.append(groupedOrder)
        if returnGroups_df:
            returnObj.append(groupedUmap)
        if len(returnObj) == 1:
            return returnObj[0]
        elif len(returnObj) > 1:
            return returnObj

    def highConfidenceErrorsByAttribute(self, barBinCol=None, compareCol=None, plotGroups=False,
                                        cmap=None, sort=True, horizontal=True,
                                        ticklabels=True, returnGroups_dict=False, returnGroups_df=False,
                                        threshold=0.45, verbose=True):
        """
        Calculate and display high confident error rates per attribute or sets of attributes.

        :param barBinCol: str, list, or list of lists. Categorical column(s) presented in bar plots.
        :param compareCol: str. Name of column by which to separate data for comparisons. Must be provided.
        :param plotGroups: Boolean (default False).
        :param cmap: str. Name of matplotlib color map.
        :param sort: Boolean (default True). Sorts the bar plot.
        :param horizontal: Boolean (default True). Horizontal or vertical bar plot.
        :param ticklabels: Boolean (default True). To display tick labels
        :param returnGroups_dict: Boolean (default False). To return dictionary of model to group mappings.
        :param returnGroups_df: Boolean (default False). To return dataframe of groupings.
        :param threshold: float (default 0.45). Value of clustering cutoff for kmeans.
        :param verbose: Boolean (optional, default True). Prints natural language description of clusters.
        """
        self._logFunc(WORKFLOW_CATEGORY, params=locals())
        if barBinCol is None:
            recall = False
            if plotGroups:
                barBinCol = self.groupModelCol
                compareOrder = self.compareOrder
                colors = self.colors
            else:
                barBinCol = self.modelCol
                compareOrder = self.modelCompareOrder
                colors = self.modelColors
        else:
            if barBinCol == self.plotsGroupbyCol:
                recall = False
                compareOrder = self.compareOrder
                colors = self.colors
            else:
                recall = True

        if compareCol is None:
            raise ValueError("Attribute column to compare must be passed to compareCol parameter.")

        barBinCol = barBinCol if isinstance(barBinCol, list) else [barBinCol] 
        groupOrders, groupDFs = {}, []
        for col in barBinCol:
            df = self.predictions.copy()
            comboCol = col
            # Create single column that combines list of columns
            if len(col) > 1 and isinstance(col, list):
                comboCol = "-".join(col)
                df[comboCol] = df[col].apply(lambda x: '-'.join(x), axis=1)
            elif len(col) == 1 and isinstance(col, list):
                comboCol = col[0]
            if recall:
                colors, compareOrder = color_prep.prep_colors_for_bars(df, comboCol, self.cmap, None)
            for i, grp in df.groupby(compareCol):
                IPython.display.display(IPython.display.HTML("<hr>"))
                IPython.display.display(
                    IPython.display.HTML(
                        f"<h3>Comparing {i} and {comboCol}</h3>"
                    )
                )
                groupedOrder, groupedUmap = self.groupModelsByPerformance(plot_type="total", modelCol=comboCol,
                                                                          df=grp, barBinCol=comboCol,
                                                                          plotGroups=plotGroups,
                                                                          overall=False, threshold=threshold)
                groupOrders[f"{i}_{comboCol}"] = groupedOrder
                groupedUmap["Compare_BarBin Cols"] = f"{i}_{comboCol}"
                groupDFs.append(groupedUmap)
                grp["GroupForPlotting"] = grp[comboCol].map(groupedOrder)
                groups = set(groupedOrder.values())
                numGroups = len(groups)
                cDescKeys = self.clusterDescriptions.keys()
                for groupName in groups:
                    if groupName not in cDescKeys:
                        self.clusterDescriptions[f"Group {groupName}"] = {}
                    subset = grp.loc[grp["GroupForPlotting"] == groupName]
                    IPython.display.display(
                        IPython.display.HTML(
                            "<h4>Illustrating the error rate, highlighting in bold the high confidence errors:</h4>"))
                    error_title = "High Confidence Errors Relative to Total Errors"
                    total_title = "High Confidence Errors Relative to Total Predictions"

                    title = total_title

                    if numGroups > 1:
                        title += f" - Group {groupName}"
                    perModel = True if numGroups > 1 else False
                    # Generate Bar plot showing high confidence errors over total errors.
                    data, compareOrder_updated = error_data(subset, comboCol, plot_type="total",
                                                            correctCol=self.correctCol,
                                                            highConfidenceCol="high confidence", perModel=perModel,
                                                            compareOrder=compareOrder, colors=colors, overall=False,
                                                            sort=True)

                    try:
                        desc_a = self.clusterDescriptions[f"Group {groupName}"]
                    except KeyError:
                        desc_a = None

                    cluster_description_a = plot_error_bars(
                        data,
                        comboCol,
                        plot_type="total",
                        compareOrder=compareOrder_updated,
                        title=f"{total_title} - Group {groupName}",
                        ticklabels=ticklabels,
                        horizontal=horizontal,
                        descName="highConfidenceErrors",
                        desc=desc_a,
                        clusterCol=comboCol,
                        verbose=verbose
                    )

                    IPython.display.display(
                        IPython.display.HTML(
                            "<h4>Comparing the relative percentage of errors that are high confidence across models:</h4>"
                        )
                    ) 
                    data, compareOrder_updated = error_data(subset, comboCol, plot_type="error",
                                                            correctCol=self.correctCol,
                                                            highConfidenceCol="high confidence", perModel=perModel,
                                                            compareOrder=compareOrder, colors=colors, overall=False,
                                                            sort=True)

                    # Generate Bar plot showing high confidence errors over total errors.
                    cluster_description_b = plot_error_bars(
                        data,
                        comboCol,
                        plot_type="error",
                        compareOrder=compareOrder_updated,
                        title=f"{error_title} - Group {groupName}",
                        ticklabels=ticklabels,
                        horizontal=horizontal,
                        descName="highConfidenceErrorsOnly",
                        desc=desc_a,
                        clusterCol=comboCol,
                        verbose=verbose
                    )
                    # Save descriptions if function called again
                    if len(desc_a.keys()) == 0:
                        self.clusterDescriptions[f"Group {groupName}"]["highConfidenceErrors"] = cluster_description_a
                        self.clusterDescriptions[f"Group {groupName}"]["highConfidenceErrorsOnly"] = cluster_description_b

        returnObj = []
        if returnGroups_dict:
            returnObj.append(groupOrders)
        if returnGroups_df:
            groupDFs = pd.concat(groupDFs)
            returnObj.append(groupDFs)
        if len(returnObj) == 1:
            return returnObj[0]
        elif len(returnObj) > 1:
            return returnObj

    def confidenceComparisons(self, plotGroups=False, returnGroups_dict=False, returnGroups_df=False, threshold=0.45):
        """
        Calculate and display cumulative and probability distributions of predicted confidence for several comparisons:
        1) Correct vs Incorrect
        2) High vs Low Confidence
        3) Gold Binary Class 1 vs Class 2
        4) Pred Binary Class 1 vs Class 2
        :param plotGroups: Boolean (default False).
        :param returnGroups_dict: Boolean (default False).
        :param returnGroups_df: Boolean (default False).
        :param threshold: float (default 0.45).
        """
        self._logFunc(WORKFLOW_CATEGORY, params=locals())

        groupedOrder, groupedUmap = self.groupModelsByPerformance(plot_type="total", modelCol=self.modelCol,
                                                                  plotGroups=plotGroups,
                                                                  threshold=threshold)
        self.predictions["GroupForPlotting"] = self.predictions[self.modelCol].map(groupedOrder)
        groups = set(groupedOrder.values())
        numGroups = len(groups)
        for groupName in groups:
            predictions = self.predictions.loc[self.predictions["GroupForPlotting"] == groupName].copy()
            IPython.display.display(
                IPython.display.HTML(f"<h3>Significantly different confidence distributions to examine</h3>")
            )

            if predictions[self.correctCol].dtype == int:
                predictions[self.correctCol] = predictions[self.correctCol].astype(bool)

            predictions["Correct"] = predictions[self.correctCol].apply(lambda x: "Correct" if x else "Incorrect")
            predictions["High Confidence"] = predictions["high confidence"].apply(lambda x: "High" if x else "Low")
            try:
                predictions["Gold"] = predictions[self.goldCol].apply(lambda x: x.title())
                predictions["Pred"] = predictions[self.predCol].apply(lambda x: x.title())
            except AttributeError:
                predictions["Gold"] = predictions[self.goldCol]
                predictions["Pred"] = predictions[self.predCol]

            compareCols = ["Correct", "High Confidence", "Gold", "Pred"]
            compareText = ["", "Confidence: ", "Gold: ", "Pred: "]

            for (test, model), mdf in predictions.groupby([self.testSetCol, self.modelCol]):
                ##
                IPython.display.display(IPython.display.HTML(f"<h4>{model} on {test}</h4>"))

                for col, text in zip(compareCols, compareText):
                    curr = stat_data(mdf, compareCol=col, confidenceCol=self.confidenceCol, round_confidence=2)
                    title = f"{model} - {text}{' vs. '.join([str(c) for c in curr.keys()])}"
                    if numGroups > 1:
                        new_title = title + " - Group " + str(groupName)
                    else:
                        new_title = title
                    stat_plot(curr, confidenceCol=self.confidenceCol, compareColor=["tab:green", "tab:pink"],
                              dist="pdf", title=new_title)
                    stat_plot(curr, confidenceCol=self.confidenceCol, compareColor=["tab:green", "tab:pink"],
                              dist="cdf", title=new_title)
                    plt.show()
        returnObj = []
        if returnGroups_dict:
            returnObj.append(groupedOrder)
        if returnGroups_df:
            returnObj.append(groupedUmap)
        if len(returnObj) == 1:
            return returnObj[0]
        elif len(returnObj) > 1:
            return returnObj

    def plotClusters(self, grp_dict, df, reduction_method="umap", original_coords=False):
        """
        Plot 2D projection of clusters.
        :param grp_dict: Dictionary. From one of failure cost functions.
        :param df: DataFrame. From one of failure cost functions.
        :param reduction_method: str. One of 'umap', 'tsne', or 'pca'.
        :param original_coords: Boolean.
        """
        self._logFunc(WORKFLOW_CATEGORY, params=locals(), removeParamFields=['df'])

        df = projection(df, self.modelCol, self.random_seed, reduction_method=reduction_method)

        plot_clusters(df, grp_dict, self.modelCol, original_coords)
