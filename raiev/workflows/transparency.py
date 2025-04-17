import IPython
import numpy as np
import pandas as pd
import networkx as nx
import seaborn as sns
import matplotlib.pyplot as plt

from nltk import everygrams
from collections import Counter

from raiev.vis import plot_clusters
from networkx.algorithms import bipartite
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import precision_recall_fscore_support
from raiev.assistance import clusterModelsByData, projection

from .base import RAIEVanalysis, cleanParams

from raiev.utils import english_stops

WORKFLOW_CATEGORY = "Model Transparency"
WORKFLOW_DESC = "Model Transparency evaluations outline exploratory error analysis workflows that expand on those described above for holistic understanding of model behavior based on input-output relationships introducing transparency (of varying degrees) to even black box systems."

WORKFLOW_OUTLINE = """## import the raiev package
import raiev
## import the Model Transparency workflow sub-module
from raiev.workflows import transparency

## Load model predictions
predictions = load_predictions(filepath_to_predictions)
    
## Instantiation of workflow object
t = transparency.analysis()

## Analysis using TExplore

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


    Analysis class for the Transparency Workflow

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
        goldCol='gold',
        predCol='pred',
        predID="id",
        predConfidence='confidence',
        textCol='text',
        goldBinaryCol=None,
        random_seed=512,
        log_savedir=None,
        loadLastLog=False,
        loadLogByName=None,
        logger=None,
        interactive=True,
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
        RAIEVanalysis.__init__(
            self,
            log_savedir,
            load_last_session=loadLastLog,
            load_session_name=loadLogByName,
            logger=logger,
            workflow_outline=WORKFLOW_OUTLINE,
            interactive=interactive,
        )
        ##

        # log initialization
        self._logFunc(WORKFLOW_CATEGORY, params=params)

        self.predictions = predictions
        self.taskType = taskType
        self.modelCol = modelCol
        self.testSetCol = testSetCol
        self.predID = predID
        self.textCol = textCol
        self.taskCol = taskCol
        self.goldCol = goldCol
        self.predCol = predCol
        self.predConfidence = predConfidence
        self.goldBinaryCol = goldBinaryCol
        self.random_seed = random_seed

    def workflow(self):
        """
        Check Default Workflow Recommendation
        """
        self._logFunc(WORKFLOW_CATEGORY, params=locals())
        workflow()

    def _draw_bipartite_graph(self, subset_df, left_classes, right_classes, left_col, right_col):
        """
        Helper function to draw bipartite graph.

        :param subset_df: (Pandas DataFrame) Data to plot.
        :param left_classes: (list) optional, default None. Names of nodes for left side.
        :param right_classes: (list) optional, default None. Names of nodes for right side.
        :param left_col: (str) optional, default None. Column that has left_classes.
        :param right_col: (str) optional, default None. Column that has right_classes.
        """
        self._logFunc(WORKFLOW_CATEGORY, params=locals(), removeParamFields=['subset_df'])
        if left_classes is None and right_classes is None:
            raise ValueError("At least one of {left_classes, right_classes} cannot be None.")

        G = nx.Graph()
        # Ngram Nodes
        nodes = subset_df["variable"].unique()

        # Edges from ngram nodes to left side nodes
        if left_classes is not None:
            left_edges = Counter(
                [
                    (x, y)
                    for x, y in zip(
                        subset_df.loc[subset_df[left_col].isin(left_classes)][left_col].tolist(),
                        subset_df["variable"].tolist(),
                    )
                ]
            )
            left_edges = [(x[0], x[1], w) for x, w in left_edges.items()]

            G.add_nodes_from(left_classes, bipartite=0)
            G.add_nodes_from(nodes, bipartite=1)
            G.add_weighted_edges_from(left_edges)

        # Edges from ngram nodes to right side nodes
        if right_classes is not None:
            right_edges = Counter(
                [
                    (y, x)
                    for x, y in zip(
                        subset_df.loc[subset_df[right_col].isin(right_classes)][right_col].tolist(),
                        subset_df["variable"].tolist(),
                    )
                ]
            )
            right_edges = [(x[0], x[1], w) for x, w in right_edges.items()]

            if left_classes is not None:
                G.add_nodes_from(right_classes, bipartite=2)
            else:
                G.add_nodes_from(nodes, bipartite=0)
                G.add_nodes_from(right_classes, bipartite=1)

            G.add_weighted_edges_from(right_edges)

        nodes = G.nodes()
        # for each of the parts create a set
        nodes_0 = set([n for n in nodes if G.nodes[n]['bipartite'] == 0])
        nodes_1 = set([n for n in nodes if G.nodes[n]['bipartite'] == 1])
        nodes_2 = set([n for n in nodes if G.nodes[n]['bipartite'] == 2])

        # set the location of the nodes for each set
        pos = dict()
        pos.update((n, (1, i)) for i, n in enumerate(nodes_0))
        pos.update((n, (2, i)) for i, n in enumerate(nodes_1))
        pos.update((n, (3, i * 5)) for i, n in enumerate(nodes_2))

        widths = nx.get_edge_attributes(G, 'weight')

        plt.figure(figsize=(15, 15))

        nx.draw_networkx_nodes(G, pos, node_color='none', linewidths=0.5, edgecolors='red', alpha=0.7)
        nx.draw_networkx_edges(
            G, pos, edgelist=widths.keys(), width=[x * 0.01 for x in widths.values()], edge_color='gray', alpha=0.5
        )
        nx.draw_networkx_labels(G, pos)

        plt.axis('off')
        plt.margins(x=0.4)
        plt.show()

    def draw_ngram_bipartite_graphs(
        self, groupbyCol=None, left_classes=None, right_classes=None, left_col=None, right_col=None
    ):
        """
        Draw bipartite graph with ngram as nodes based on classes.

        :param groupbyCol: (str) optional, default None.
                Create bipartition based on classes in groupbyCol, else use all.
        :param left_classes: (list) optional, default None. Names of nodes for left side.
        :param right_classes: (list) optional, default None. Names of nodes for right side.
        :param left_col: (str) optional, default None. Column that has left_classes.
        :param right_col: (str) optional, default None. Column that has right_classes.
        """
        self._logFunc(WORKFLOW_CATEGORY, params=locals())

        if left_col is None:
            left_col = self.goldCol
        if right_col is None:
            right_col = self.goldBinaryCol

        lst = list(self.predictions[self.goldCol].unique()) + list(self.predictions[self.predCol].unique())
        ignore_cols = [
            self.modelCol,
            self.taskCol,
            self.testSetCol,
            self.predConfidence,
            self.textCol,
            'split',
            'Ngrams',
        ] + lst
        check_if_exists = ['trainset', 'experiment', 'correct', 'gold-rank', 'date']
        for c in check_if_exists:
            if c in self.predictions.columns:
                ignore_cols.append(c)

        if left_col == right_col:
            id_var_cols = [left_col]
        else:
            id_var_cols = [left_col, right_col]

        if groupbyCol is not None:
            keeps = [x for x in self.predictions.columns if x not in ignore_cols]
            bipartite_df = pd.melt(
                self.predictions.loc[self.predictions[self.taskCol] == groupbyCol],
                id_vars=id_var_cols,
                value_vars=keeps,
            )
            contains_df = bipartite_df.loc[bipartite_df['value'] == "contains"]
            self._draw_bipartite_graph(contains_df, left_classes, right_classes, left_col, right_col)
        else:
            keeps = [x for x in self.predictions.columns if x not in ignore_cols]
            bipartite_df = pd.melt(self.predictions, id_vars=id_var_cols, value_vars=keeps)
            contains_df = bipartite_df.loc[bipartite_df['value'] == "contains"]
            self._draw_bipartite_graph(contains_df, left_classes, right_classes, left_col, right_col)

    def cluster_instances(
        self,
        valueCol,
        threshold=0.45,
        reduction_method="umap",
        original_coords=False,
        hueCol="Cluster",
        numClusters=None,
        plotInertia=False,
    ):
        """
        Cluster instances using K-Means; plot after dimensionality reduction.

        :param valueCol: (str) column in predictions dataframe to cluster on
        :param threshold: (float) optional, default 0.45
        :param reduction_method: (str) optional, default umap. Must be one of umap, tsne, pca, lle.
        :param original_coords: (bool) optional, default False
        :param hueCol: (str) optional, default group. Column to color clusters by.
        :param numClusters: (int) optional, default None. If specified, uses this number of clusters for kmeans.
                                Ignores threshold parameter.
        :param plotInertia: (bool) optiona, default False. If True, displays the inertia plot from kmeans.
        """
        self._logFunc(WORKFLOW_CATEGORY, params=locals())

        # Check that each model in self.modelCol has equal number of rows (must be equal for kmeans and dim reduction)
        num_instances = self.predictions[self.modelCol].value_counts().nunique()
        if num_instances != 1:
            # If not equal, reduce until they are equal
            num_models = self.predictions[self.modelCol].nunique()
            t = self.predictions.groupby([self.predID], as_index=False)[self.modelCol].count()
            t = t.loc[t[self.modelCol] == num_models]
            self.predictions = self.predictions.loc[
                self.predictions[self.predID].isin(t[self.predID].tolist())
            ].reset_index(drop=True)

        pivot_df = (
            self.predictions.groupby(self.modelCol)
            .agg({valueCol: list})
            .rename(columns={valueCol: "y"})
            .reset_index(drop=False)
        )
        # Add hueCol back if not "group"
        if hueCol not in pivot_df.columns and hueCol in self.predictions.columns:
            pivot_df[hueCol] = pivot_df[self.modelCol].map(self.predictions.set_index(self.modelCol)[hueCol].to_dict())

        # Cluster with kmeans
        data = clusterModelsByData(
            pivot_df, self.modelCol, self.random_seed, threshold, numClusters=numClusters, plotInertia=plotInertia
        )
        groupOrder = data.set_index(self.modelCol)["Cluster"].to_dict()
        # Dimension reduction
        df = projection(data, self.modelCol, random_seed=self.random_seed, reduction_method=reduction_method)

        # Plot clusters
        plot_clusters(df, groupOrder, self.modelCol, original_coords=original_coords, hueCol=hueCol)

        return groupOrder, df.set_index(self.modelCol)
