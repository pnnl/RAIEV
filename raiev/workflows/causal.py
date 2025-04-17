import IPython
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tqdm
import networkx as nx

from ..causal import (
    processCategoricalCols,
    processDate,
    createConfusionMatrix,
    createTextFactors,
    createPredictionCols,
)
from ..causal import causalDiscovery, parseCausalCmd, correctDirections, connectCategoricals
from ..causal import drawCausalGraph, get_colliders, get_mediators
from ..causal import (
    isBinary,
    binaryTreatmentEffectEstimates,
    continuousTreatmentEffectEstimates,
    setupAxes,
    plotContextContinuous,
    plottingContinuous,
)

WORKFLOW_CATEGORY = "Causal Informed Insights" 
WORKFLOW_DESC = "Causal Informed Insights evaluations provide insights on causal relationships affecting model performance, including insights that can inform performance on unseen or underrepresented circumstances. Evaluation workflow combines causal structural learning and treatment effect estimation methodology."

WORKFLOW_OUTLINE = """## import the raiev package
import raiev
## import the causal workflow sub-module
from raiev.workflows import causal

## Load model predictions
predictions = load_predictions(filepath_to_predictions)

## Instantiation of workflow object
acc = causal.analysis(predictions,
                      task='classification',
                      prediction_col = 'pred',
                      label_col = 'gold',
                      positive_value='Alert',
                      date_col = 'date',
                      text_col = 'text',
                      categorical_cols = ['Label Class','Subset of Test Category'],
                      mode='predictions')

## Causal discovery
acc.causal_discovery()

## Causal inference
acc.causal_inference()
"""


def workflow():
    """Display example workflow code for Accountability Overviews."""
    IPython.display.display(IPython.display.HTML(f"<h3>Overview of {WORKFLOW_CATEGORY} Workflow</h3>"))
    print(WORKFLOW_DESC)
    IPython.display.display(IPython.display.HTML("<b>Implemented workflow includes:</b>"))
    print(WORKFLOW_OUTLINE)


class analysis:
    """ 
    ------------------------------------------
    
    
    Analysis class for the Causal Informed Insights Workflow
    
    Want a recommendation of analysis functions to explore next? Try calling the ___ .suggestWorkflow() ___ method!
    

    ------------------------------------------
    """

    def __init__(
        self,
        predictions,
        task='classification',
        prediction_col='pred',
        label_col='gold',
        confidence_col=None,
        positive_value=1,
        date_col=None,
        text_col=None,
        categorical_cols=[],
        factor_cols=[],
        outcome_cols=[],
        other_cols=[],
        mode='confusion_matrix',
        verbosity=1,
    ):

        if task == 'regression':
            assert mode in [
                'errors',
                'percent_errors',
            ], "Error! Invalid task and mode combination. Mode for regression task must be 'errors' or 'percent_errors'."
        if task == 'classification':
            assert mode in [
                'confusion_matrix',
                'predictions',
            ], "Error! Invalid task and mode combination. Mode for regression task must be either 'confusion_matrix' or 'predictions'"
        if task == 'classification' and isinstance(positive_value, list):
            assert mode in [
                'predictions',
            ], "Error! Invalid task and mode combination. Mode for multiclass classification must be 'predictions'"

        self.predictions = predictions.copy()

        self.prediction_col = prediction_col
        self.label_col = label_col
        self.positive_value = positive_value
        self.date_col = date_col
        self.categorical_cols = categorical_cols
        self.text_col = text_col
        self.confidence_col = confidence_col
        self.verbosity = verbosity

        self.process_data(task)

        factor_cols += [c for c in self.predictions.columns if 'factor' in c.lower()]
        factor_cols = list(set(factor_cols))

        if len(outcome_cols) == 0 and mode == 'confusion_matrix':
            outcome_cols = [c for c in self.predictions.columns if self._check_cm_col(c)]
        elif len(outcome_cols) == 0 and mode == 'predictions':
            outcome_cols = [c for c in self.predictions.columns if self._check_pred_col(c)]
        elif mode == 'errors':
            outcome_cols = ['Outcome.Error', 'Outcome.Prediction']
        elif mode == 'percent_errors':
            outcome_cols = ['Outcome.Percent_Error', 'Outcome.Prediction']
 
        outcome_cols = list(set(outcome_cols))

        if (
            task == 'classification'
            and not isinstance(positive_value, list)
            and f'{positive_value} Ground Truth' not in other_cols
        ):
            other_cols += [f'{positive_value} Ground Truth']
        elif task == 'regression' and 'Ground Truth' not in other_cols:
            other_cols += ['Ground Truth']
        elif task == 'classification' and isinstance(positive_value, list):
            for pos_val in positive_value:
                if pos_val not in other_cols:
                    other_cols += [f'{pos_val} Ground Truth']

        self.col_types = {'Outcomes': outcome_cols, 'Factors': factor_cols, 'Other': other_cols}

        self.causal_columns = factor_cols + outcome_cols + other_cols

        self._vprint('Using the following variables for causal analysis...', 1)
        for col in factor_cols:
            self._vprint(f'\t{col} (Factor)', 1)
        for col in outcome_cols:
            self._vprint(f'\t{col} (Outcome)', 1)
        if len(other_cols) > 0:
            for col in other_cols:
                self._vprint(f'\t{col} (Other)', 1)

        self.causal_graph = None

    def _vprint(self, text, verbosity_setting):
        """Method to print if verbosity is above verbosity_setting."""
        if self.verbosity >= verbosity_setting:
            print(text)

    def _check_cm_col(self, col):
        """Method to check if col is a confusion matrix column."""

        col = col.lower()

        if ('true' in col or 'false' in col) and ('pos' in col or 'neg' in col):
            return True

        if col in ['tp', 'fp', 'tn', 'fn']:
            return True

        return False

    def _check_pred_col(self, col):
        """Method to check if col is a pred-type column."""

        if 'Outcome.Correct' in col or 'Outcome.Predict' in col:
            return True

        return False

    def process_data(self, task):
        """Create standard set of Factor and Outcome columns for causal discovery and inference."""
        self.predictions = processCategoricalCols(self.predictions, self.categorical_cols)

        if self.date_col is not None:
            self.predictions = processDate(self.predictions, self.date_col)

        if task == 'classification':
            if not isinstance(self.positive_value, list):
                self.predictions = createConfusionMatrix(
                    self.predictions, self.label_col, self.prediction_col, self.positive_value
                )

            self.predictions = createPredictionCols(
                self.predictions,
                self.label_col,
                self.prediction_col,
                self.positive_value,
                confidence_col=self.confidence_col,
            )

            if not isinstance(self.positive_value, list):
                ground_truth_cols = [self.positive_value]
            else:
                ground_truth_cols = self.positive_value

            for col in ground_truth_cols:
                self.predictions[f'{col} Ground Truth'] = self.predictions[self.prediction_col] == col
        else: # Regression
            self.predictions['Outcome.Error'] = (
                (self.predictions[self.prediction_col] - self.predictions[self.label_col]).abs().astype(float)
            )

            if len((self.predictions[self.label_col] == 0.0) & (self.predictions[self.prediction_col] == 0.0)) > 0:
                epsilon = 0.01 * self.predictions[self.label_col].std()
                denom = (
                    0.5 * (self.predictions[self.label_col].abs() + self.predictions[self.prediction_col].abs())
                    + epsilon
                )
                print(
                    f"Warning: datapoints with both label and prediction as zero exist, so relative percent difference with epsilon={epsilon:.2f} being used rather than percent error"
                )
            elif 0.0 in self.predictions[self.label_col]:
                denom = 0.5 * (self.predictions[self.label_col].abs() + self.predictions[self.prediction_col].abs())
                print(
                    "Warning: labels contain zeros, so relative percent difference being used rather than percent error"
                )
            else:
                denom = self.predictions[self.label_col].abs()

            self.predictions['Outcome.Percent_Error'] = 100.0 * self.predictions['Outcome.Error'] / denom

            self.predictions['Outcome.Prediction'] = self.predictions[self.prediction_col]
            self.predictions['Ground Truth'] = self.predictions[self.label_col]

        if self.text_col is not None:
            self.predictions = createTextFactors(self.predictions, self.text_col)

    def causal_discovery(self, jar_path='./', path='./', ident='model_1', pd=3.0):
        """
        Perform causal discovery to identify relationships between data factors and model prediction outcomes.

        :param jar_path (str), default "./". path to the causal-cmd jar file.
        :param path: (str) optional, default "./". path to write output files that are created by the causal discovery algorithm
        :param ident: (str) identify to use in output files created by the causal discovery algorithm
        :param pd: (float) optional, default 3.0. Penalty discount, a parameter of the BOSS algorithm. Higher values will decrease the complexity of the graph.
        """
        cd_output_file = causalDiscovery(
            self.predictions[self.causal_columns], output_path=path, jar_path=jar_path, ident=ident, pd=pd
        )
        self.causal_graph = parseCausalCmd(cd_output_file)
        self.causal_graph = correctDirections(self.causal_graph)

        self.causal_graph = connectCategoricals(self.causal_graph)

        self.nx_causal_graph = nx.from_pandas_edgelist(
            self.causal_graph, source="cause", target="effect", create_using=nx.DiGraph
        )

        drawCausalGraph(self.causal_graph, col_types=self.col_types)

    def causal_inference(
        self,
        cause=None,
        effect=None,
        edgelist=None,
        confounders=[],
        force=False,
        context_plot_col=None,
        confounding_sensitivity=False,
    ):
        """
        Perform causal inference to identify the direction and size of the causal relationships between data factors and model prediction outcomes.

        By default this will detect causal effects of all Factor -> Outcome edges in the inferred causal graph from the causal_discovery function.
        You can also specify a specific pair of variables (using the cause and effect parameters) or a specific list of variable pairs (using the edgelist parameter)

        :param cause: (str) optional, specify a specific variable to use as the cause in the causal inference
        :param effect: (str) optional, specify a specific variable to use as the effect in the causal inference
        :param edgelist: (str) optional, provide a dataframe with "cause" and "effect" columns to specify a list of specific variable combinations to use for causal inference
        :param confounders: (list) optional, specify a list of confounders to control for. if not provided then all variables other than mediators and colliders will be used.
        :param force: (boolean) optional, default = False, if True, use the specified confounder list without removing mediators and colliders
        :param context_plot_col: (str) optional, a column in the data to plot along with the causal effects to provide context
        :param confounding_sensitivity: (boolean) optional, default = False, randomly modify the confounder list to observe whether the results are robust to the confounder selection
        """
        if cause is not None and effect is not None:
            print(f'Detecting causal of effect of specified edge ({cause} -> {effect})')
            assert (cause in self.predictions.columns) and (
                effect in self.predictions.columns
            ), f"Error! {cause} and/or {effect} not in data."
            edges_subset = pd.DataFrame({'cause': [cause], 'effect': [effect]})
        elif edgelist is not None:
            print('Detecting causal effects of edges in:')
            print(edgelist)
            assert ('cause' in edgelist.columns) and (
                'effect' in edgelist.columns
            ), "Error! edgelist needs to have 'cause' and 'effect' columns."
            edges_subset = edgelist.copy()
        elif self.causal_graph is not None:
            print('Detecting causal effects of all Factor -> Outcome edges in the inferred causal graph:')
            edges_subset = self.causal_graph[
                (self.causal_graph['cause'].str.startswith('Factor.'))
                & (self.causal_graph['effect'].str.startswith('Outcome.'))
            ]
            print(edges_subset)
        else:
            raise Exception("No valid cause/effect combinations provided for inference.")

        if len(confounders) == 0:
            confounders = self.causal_columns

        binary_results = []

        for e, edge in edges_subset.iterrows():

            self._vprint('', 1)
            self._vprint('-----' * 20, 1)
            cause = edge['cause']
            effect = edge['effect']

            all_confounder_list = [c for c in confounders if c != cause and c != effect]
            confounder_list = [c for c in confounders if c != cause and c != effect]

            if not force:
                colliders = get_colliders(self.nx_causal_graph, cause, effect)
                mediators = get_mediators(self.nx_causal_graph, cause, effect)

                self._vprint(f'Cause: {cause}, Effect: {effect}', 1)

                self._vprint('', 2)
                self._vprint('Analyzing causal graph to remove colliders and mediators from the confounder list...', 1)
                self._vprint('', 2)
                if len(colliders) > 0:
                    self._vprint('\tRemoving colliders from confounder list...', 1)
                    for collider in colliders:
                        self._vprint(f'\t\t{collider}', 2)
                        if collider in confounder_list:
                            confounder_list.remove(collider)
                else:
                    self._vprint('\tNo colliders detected.', 2)

                self._vprint('', 2)
                if len(mediators) > 0:
                    self._vprint('\tRemoving mediators from confounder list...', 1)
                    for mediator in mediators:
                        self._vprint(f'\t\t{mediator}', 2)
                        if mediator in confounder_list:
                            confounder_list.remove(mediator)
                else:
                    self._vprint('\tNo mediators detected.', 2)

            self._vprint('', 1)
            self._vprint(f'Detecting causal effect of {cause} on {effect} controlling for confounders...', 1)
            for confounder in confounder_list:
                self._vprint(f'\t{confounder}', 2)
            self._vprint('', 1)

            if confounding_sensitivity:
                confounding_combos = []

                confounding_combos = [confounder_list]
                count = 0
                while len(confounding_combos) < 10 and count < 100:
                    rand = np.random.choice(np.array([0, 1]))
                    if rand == 1 and len(confounder_list) < len(all_confounder_list):
                        extra_cols = [c for c in all_confounder_list if c not in confounder_list]
                        to_add = confounder_list + list(np.random.choice(extra_cols, 1))
                        if to_add not in confounding_combos:
                            confounding_combos.append(to_add)
                    else:
                        to_remove = np.random.choice(confounder_list)

                        removed = [c for c in confounder_list if c != to_remove]

                        if removed not in confounding_combos and len(removed) > 0:
                            confounding_combos.append(removed)

                    count += 1

            else:
                confounding_combos = [confounder_list]

            if len(confounding_combos) > 1:
                loop = tqdm.tqdm(confounding_combos)
            else:
                loop = confounding_combos

            if isBinary(self.predictions[cause]):

                valid = True
                for g, grp in self.predictions.groupby(cause):
                    if grp[effect].nunique() < 2:
                        self._vprint('Only 1 outcome class in one of the treatment groups. Skipping...', 1)
                        valid = False

                if not valid:
                    break

                for confounds in loop:
                    results = binaryTreatmentEffectEstimates(self.predictions, cause, effect, confounds)
                    binary_results.append(results)
            else:
                sns.set_context("talk")
                axs = setupAxes(context_plot_col)

                min = None
                max = None

                for confounds in loop:
                    results, effect_col = continuousTreatmentEffectEstimates(
                        self.predictions, cause, effect, confounds
                    )

                    if results is None:
                        continue

                    min = results['Treatment'].min()
                    max = results['Treatment'].max()

                    plottingContinuous(results, cause, effect, effect_col, axs=axs)

                plotContextContinuous(
                    self.predictions, cause, axs=axs, context_plot_col=context_plot_col, min=min, max=max
                )

                plt.show()

        if len(binary_results) > 0:

            if isBinary(self.predictions[effect]):
                metric = 'odds-ratio-increase'
            else:
                metric = 'risk-difference'

            binary_results = pd.DataFrame(binary_results)

            binary_results = binary_results.sort_values(['outcome', metric])

            binary_results['outcome'] = binary_results['outcome'].str.replace('Outcome.', '')
            binary_results['treatment'] = binary_results['treatment'].str.replace('Factor.', '')

            for g, grp in binary_results.groupby('outcome'):

                sig_results = []
                colors = {}
                for g2, grp2 in grp.groupby('treatment'):
                    if len(grp2) > 1:
                        ci = sns.utils.ci(sns.algorithms.bootstrap(grp2[metric]))

                        if ci[0] * ci[1] > 0:
                            colors[g2] = sns.color_palette()[0]
                            sig_results.append([g2, g])
                        else:
                            colors[g2] = sns.color_palette()[1]
                    else:
                        colors[g2] = sns.color_palette()[0]

                grp['color'] = grp['treatment'].map(colors)

                ax = sns.barplot(data=grp, x='treatment', y=metric, hue='color')
                ax.legend_.remove()
                plt.xticks(rotation=90)
                plt.title(f'Treatment Effect on {g}')
                plt.xlabel('Treatment')
                if metric == 'risk-difference':
                    plt.ylabel('Difference Between\nTreated and Untreated')
                if metric == 'odds-ratio-increase':
                    plt.ylabel('Odds Ratio Increase\nof Treated Group')

                plt.show()

                if len(sig_results) > 0:
                    self._vprint('Causal relationships robust to confounder sampling:', 1)
                for sig_result in sig_results:
                    self._vprint(f'\t{sig_result[0]} -> {sig_result[1]}', 1)