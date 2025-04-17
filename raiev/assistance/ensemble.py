import statistics
import numpy as np
import pandas as pd


def make_ensemble_model(predictions, ensemble_method, correctCol, goldCol, modelCol, predCol, predID, predConfidence,
                        positiveLabel, negativeLabel):
    def _get_avg_conf(row):
        class_label = row["ens_" + ensemble_method]
        avg = []
        for cls, conf in zip(row[predCol], row[predConfidence]):
            if cls == class_label:
                avg.append(conf)
        return statistics.mean(avg)

    def _ensemble_weighted_confidence(row):
        # Normalize confidences between 0 and 1
        raw = row[predConfidence]
        norm = [float(i) / sum(raw) for i in raw]
        class_weights = {c: [] for c in set(row[predCol])}
        # Average confidences per class
        for i, p in enumerate(row[predCol]):
            class_weights[p].append(norm[i])
        for k, v in class_weights.items():
            class_weights[k] = statistics.mean(v)

        # Return the class with the highest avg. confidence
        max_class = max(class_weights, key=class_weights.get)
        return max_class, class_weights[max_class]

    def _ensemble_max_confidence(row):
        max_conf_idx = np.argmax(row[predConfidence])
        return row[predCol][max_conf_idx], row[predConfidence][max_conf_idx]

    def _set_labels(row):
        label = row[predCol]
        if label == positiveLabel:
            return row[predConfidence], 1 - row[predConfidence]
        if label == negativeLabel:
            return 1 - row[predConfidence], row[predConfidence]

    def _outcome_type(row):
        if row[goldCol] == positiveLabel and row[predCol] == positiveLabel:
            return "True Positive", True
        elif row[goldCol] == negativeLabel and row[predCol] == positiveLabel:
            return "False Positive", False
        elif row[goldCol] == positiveLabel and row[predCol] == negativeLabel:
            return "False Negative", False
        else:
            return "True Negative", True

    ensembled = pd.pivot_table(predictions, values=[predCol, predConfidence, modelCol],
                               index=[predID], aggfunc=list).reset_index(drop=False)

    if ensemble_method == "max_count":
        ensembled["ens_" + ensemble_method] = ensembled[predCol].apply(
            lambda x: max(set(x), key=x.count))  # If tie, goes with first alphabetically
        ensembled[predConfidence] = ensembled.apply(_get_avg_conf, axis=1)
    elif ensemble_method == "max_weighted_confidence":
        ensembled[["ens_" + ensemble_method, predConfidence]] = ensembled.apply(
            _ensemble_weighted_confidence, axis=1, result_type="expand")
    elif ensemble_method == "max_confidence":
        ensembled[["ens_" + ensemble_method, predConfidence]] = ensembled.apply(_ensemble_max_confidence, axis=1,
                                                                                result_type='expand')
    else:
        # Default for now
        ensembled[["ens_" + ensemble_method, predConfidence]] = ensembled.apply(_ensemble_max_confidence, axis=1,
                                                                                result_type='expand')

    temp = predictions.drop(columns=[modelCol, predCol, predConfidence])
    temp_cols_to_drop = ["Outcome", correctCol, "trainset", "pred_id", "model_architecture",
                         positiveLabel, negativeLabel, ]
    cols_to_keep = [c for c in temp.columns if c not in temp_cols_to_drop]
    temp = temp[cols_to_keep].drop_duplicates()

    ensembled = pd.merge(ensembled[[predID, "ens_" + ensemble_method, predConfidence]],
                         temp, on=predID)

    ensembled = ensembled.rename(columns={"ens_" + ensemble_method: predCol})
    ensembled[modelCol] = "ens_" + ensemble_method
    ensembled[[positiveLabel, negativeLabel]] = ensembled.apply(_set_labels, axis=1, result_type='expand')

    # Recalculate based on ensemble model
    if "Outcome" in predictions.columns and correctCol in predictions.columns:
        ensembled[["Outcome", correctCol]] = ensembled.apply(_outcome_type, axis=1, result_type='expand')

    return pd.concat([predictions, ensembled])
