import IPython
import statistics
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter
import warnings
from .base import RAIEVanalysis, cleanParams

warnings.filterwarnings('ignore')

WORKFLOW_CATEGORY = "Ensemble"
WORKFLOW_DESC = ""

WORKFLOW_OUTLINE = """## import the raiev package
import raiev
## import the ensemble workflow sub-module
from raiev.workflows import ensemble

## Load model predictions
predictions = load_predictions(filepath_to_predictions)

## Instantiation of workflow object
ens = ensemble.analysis(predictions, ensemble_method="max_count",
                        modelCol="model_alias", testSetCol="testset", goldCol="gold", predCol="pred",
                        predConfidence="confidence", predID="id", correctCol="correct",
                        positiveLabel="Alert", negativeLabel="No Alert")

## Continued Analysis, Analytic Interface(s), etc. under development
"""


def workflow():
    """"""
    outline = """## import the raiev package
import raiev
## import the Ensemble workflow sub-module
from raiev.workflows import ensemble

## Load model predictions
predictions = load_predictions(filepath_to_predictions)

## Instantiation of workflow object
es = ensemble.analysis(predictions)

## Continued Analysis, Analytic Interface(s), etc. under development

"""
    IPython.display.display(IPython.display.HTML(f"<h3>Overview of {WORKFLOW_CATEGORY} Workflow</h3>"))
    print(WORKFLOW_DESC)
    IPython.display.display(IPython.display.HTML("<b>Implemented workflow includes:</b>"))
    print(outline)


class analysis(RAIEVanalysis):
    def __init__(self,
                 predictions,
                 ensemble_method="max_count",
                 modelCol="model_alias",
                 testSetCol="dataset",
                 goldCol="gold",
                 predCol="pred",
                 predConfidence="confidence",
                 predID="id",
                 correctCol="correct",
                 positiveLabel=None,
                 negativeLabel=None,
                 taskType="binary",
                 log_savedir=None,
                 loadLastLog=False,
                 loadLogByName=None,
                 logger=None,
                 interactive=True
                 ):

        # logging prep
        # pull copy of parameters
        params = locals()
        params = cleanParams(params, removeFields=['self', 'predictions'])
        ##

        # logging
        RAIEVanalysis.__init__(self, log_savedir, load_last_session=loadLastLog, load_session_name=loadLogByName,
                               logger=logger,
                               workflow_outline=WORKFLOW_OUTLINE, interactive=interactive)
        ##

        # log initialization
        # self.logger.log(WORKFLOW_CATEGORY, 'Initializing Analysis Object', params, comment=None)
        self._logFunc(WORKFLOW_CATEGORY, params=params)
        self.predictions = predictions
        self.ensemble_method = ensemble_method
        self.modelCol = modelCol
        self.testSetCol = testSetCol
        self.goldCol = goldCol
        self.predCol = predCol
        self.predConfidence = predConfidence
        self.predID = predID
        self.correctCol = correctCol
        self.positiveLabel = positiveLabel
        self.negativeLabel = negativeLabel
        self.taskType = taskType

        def _get_avg_conf(row):
            class_label = row["ens_" + self.ensemble_method]
            avg = []
            for cls, conf in zip(row[self.predCol], row[self.predConfidence]):
                if cls == class_label:
                    avg.append(conf)
            return statistics.mean(avg)

        def _ensemble_weighted_confidence(row):
            # Normalize confidences between 0 and 1
            raw = row[self.predConfidence]
            norm = [float(i) / sum(raw) for i in raw]
            class_weights = {c: [] for c in set(row[self.predCol])}
            # Average confidences per class
            for i, p in enumerate(row[self.predCol]):
                class_weights[p].append(norm[i])
            for k, v in class_weights.items():
                class_weights[k] = statistics.mean(v)

            # Return the class with the highest avg. confidence
            max_class = max(class_weights, key=class_weights.get)
            return max_class, class_weights[max_class]

        def _ensemble_max_confidence(row):
            max_conf_idx = np.argmax(row[self.predConfidence])
            return row[self.predCol][max_conf_idx], row[self.predConfidence][max_conf_idx]

        def _set_labels(row):
            label = row[self.predCol]
            if label == self.positiveLabel:
                return row[self.predConfidence], 1 - row[self.predConfidence]
            if label == self.negativeLabel:
                return 1 - row[self.predConfidence], row[self.predConfidence]

        def _outcome_type(row):
            if row[self.goldCol] == self.positiveLabel and row[self.predCol] == self.positiveLabel:
                return "True Positive", True
            elif row[self.goldCol] == self.negativeLabel and row[self.predCol] == self.positiveLabel:
                return "False Positive", False
            elif row[self.goldCol] == self.positiveLabel and row[self.predCol] == self.negativeLabel:
                return "False Negative", False
            else:
                return "True Negative", True

        ensembled = pd.pivot_table(self.predictions, values=[self.predCol, self.predConfidence, self.modelCol],
                                   index=[self.predID], aggfunc=list).reset_index(drop=False)
        self.ensembled = ensembled

        if self.ensemble_method == "max_count":
            ensembled["ens_" + self.ensemble_method] = ensembled[self.predCol].apply(
                lambda x: max(set(x), key=x.count))  # If tie, goes with first alphabetically
            ensembled[self.predConfidence] = ensembled.apply(_get_avg_conf, axis=1)
        elif self.ensemble_method == "max_weighted_confidence":
            ensembled[["ens_" + self.ensemble_method, self.predConfidence]] = ensembled.apply(
                _ensemble_weighted_confidence, axis=1, result_type="expand")
        elif self.ensemble_method == "max_confidence":
            ensembled[["ens_" + self.ensemble_method, self.predConfidence]] = ensembled.apply(_ensemble_max_confidence,
                                                                                              axis=1,
                                                                                              result_type='expand')
        else:
            # Default for now
            ensembled[["ens_" + self.ensemble_method, self.predConfidence]] = ensembled.apply(_ensemble_max_confidence,
                                                                                              axis=1,
                                                                                              result_type='expand')

        temp = self.predictions.drop(columns=[self.modelCol, self.predCol, self.predConfidence])
        temp_cols_to_drop = ["Outcome", self.correctCol, "trainset", "pred_id", "model_architecture",
                             self.positiveLabel, self.negativeLabel]
        cols_to_keep = [c for c in temp.columns if c not in temp_cols_to_drop]
        temp = temp[cols_to_keep].drop_duplicates()

        ensembled = pd.merge(ensembled[[self.predID, "ens_" + self.ensemble_method, self.predConfidence]],
                             temp, on=self.predID)

        ensembled = ensembled.rename(columns={"ens_" + self.ensemble_method: self.predCol})
        ensembled[self.modelCol] = "ens_" + self.ensemble_method
        ensembled[[self.positiveLabel, self.negativeLabel]] = ensembled.apply(_set_labels, axis=1, result_type='expand')

        if self.taskType == "binary":
            # Recalculate based on ensemble model
            if "Outcome" in self.predictions.columns and self.correctCol in self.predictions.columns:
                ensembled[["Outcome", self.correctCol]] = ensembled.apply(_outcome_type, axis=1, result_type='expand')
        # TODO: Multiclass
        # elif self.taskType == 'multiclass':
        
        self.predictions = pd.concat([self.predictions, ensembled])

    def workflow(self):
        """
        Check Default Workflow Recommendation
        """
        self._logFunc(WORKFLOW_CATEGORY, params=locals())
        workflow()

    def diversityOfPredictions(self, attribute_cols):
        """
        :param attribute_cols: List of columns in self.predictions
        """
        self._logFunc(WORKFLOW_CATEGORY, params=locals())
        ensembled = self.predictions.loc[self.predictions[self.modelCol] == "ens_" + self.ensemble_method]
        df = self.predictions.loc[self.predictions[self.modelCol] != "ens_" + self.ensemble_method]

        num_predictions = df[self.predID].nunique()
        all_correct, all_incorrect = [], []
        even_split, maj_correct, maj_incorrect = [], [], []
        for i, grp in df.groupby(self.predID):
            grp[self.correctCol] = grp[self.correctCol].astype(int)
            length = len(grp)
            preds = list(grp[self.correctCol].unique())
            freq = Counter(grp[self.correctCol].tolist())

            if len(preds) == 1 and preds[0] == 0:
                all_incorrect.append(i)
            elif len(preds) == 1 and preds[0] == 1:
                all_correct.append(i)
            else:
                num_correct = freq[1]
                num_incorrect = freq[0]
                if (length - num_correct) / length > 0.55:
                    maj_incorrect.append(i)
                elif (length - num_incorrect) / length > 0.55:
                    maj_correct.append(i)
                else:
                    even_split.append(i)

        correct_df = df.loc[df[self.predID].isin(all_correct)]
        incorrect_df = df.loc[df[self.predID].isin(all_incorrect)]
        maj_correct_df = df.loc[df[self.predID].isin(maj_correct)]
        maj_incorrect_df = df.loc[df[self.predID].isin(maj_incorrect)]
        even_df = df.loc[df[self.predID].isin(even_split)]

        for a in attribute_cols:
            incorrect_temp = incorrect_df[a].value_counts().to_frame().reset_index(drop=False).rename(
                columns={a: "Count", 'index': a})
            incorrect_temp['Result'] = "Incorrect"
            correct_temp = correct_df[a].value_counts().to_frame().reset_index(drop=False).rename(
                columns={a: "Count", 'index': a})
            correct_temp['Result'] = "Correct"
            maj_incorrect_temp = maj_incorrect_df[a].value_counts().to_frame().reset_index(drop=False).rename(
                columns={a: "Count", 'index': a})
            maj_incorrect_temp['Result'] = "Maj. Incorrect"
            maj_correct_temp = maj_correct_df[a].value_counts().to_frame().reset_index(drop=False).rename(
                columns={a: "Count", 'index': a})
            maj_correct_temp['Result'] = "Maj. Correct"
            even_temp = even_df[a].value_counts().to_frame().reset_index(drop=False).rename(
                columns={a: "Count", 'index': a})
            even_temp['Result'] = "Even"
            ens_df = ensembled.groupby([self.correctCol, a], as_index=False).size()
            ens_df["Percentage"] = 100 * (ens_df["size"]/num_predictions)

            plot_df = pd.concat([correct_temp, maj_correct_temp, even_temp, maj_incorrect_temp, incorrect_temp])
            plot_df["Percentage"] = 100 * (plot_df["Count"]/num_predictions)
            fig, axes = plt.subplots(1, 2, figsize=(10,7), sharey=True)
            sns.barplot(data=plot_df, y=a, x="Percentage", hue="Result", ax=axes[0])
            sns.barplot(data=ens_df, y=a, x='Percentage', hue=self.correctCol, ax=axes[1])
            # plt.xscale('log')
            plt.subplots_adjust()
            axes[0].set_title("Individual Models")
            axes[1].set_title("Ensemble Model")
            axes[0].set_xlabel("Percent of Total Predictions")
            axes[1].set_xlabel("Percent of Total Predictions")
            plt.show()
