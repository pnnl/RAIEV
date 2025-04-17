"""utils.py - RAIEv Utilities."""
import os
import IPython
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from nltk import everygrams

# from nltk import download
# from nltk.corpus import stopwords
# download("stopwords")
# english_stops = stopwords.words("english")

english_stops = ['i',
                 'me',
                 'my',
                 'myself',
                 'we',
                 'our',
                 'ours',
                 'ourselves',
                 'you',
                 "you're",
                 "you've",
                 "you'll",
                 "you'd",
                 'your',
                 'yours',
                 'yourself',
                 'yourselves',
                 'he',
                 'him',
                 'his',
                 'himself',
                 'she',
                 "she's",
                 'her',
                 'hers',
                 'herself',
                 'it',
                 "it's",
                 'its',
                 'itself',
                 'they',
                 'them',
                 'their',
                 'theirs',
                 'themselves',
                 'what',
                 'which',
                 'who',
                 'whom',
                 'this',
                 'that',
                 "that'll",
                 'these',
                 'those',
                 'am',
                 'is',
                 'are',
                 'was',
                 'were',
                 'be',
                 'been',
                 'being',
                 'have',
                 'has',
                 'had',
                 'having',
                 'do',
                 'does',
                 'did',
                 'doing',
                 'a',
                 'an',
                 'the',
                 'and',
                 'but',
                 'if',
                 'or',
                 'because',
                 'as',
                 'until',
                 'while',
                 'of',
                 'at',
                 'by',
                 'for',
                 'with',
                 'about',
                 'against',
                 'between',
                 'into',
                 'through',
                 'during',
                 'before',
                 'after',
                 'above',
                 'below',
                 'to',
                 'from',
                 'up',
                 'down',
                 'in',
                 'out',
                 'on',
                 'off',
                 'over',
                 'under',
                 'again',
                 'further',
                 'then',
                 'once',
                 'here',
                 'there',
                 'when',
                 'where',
                 'why',
                 'how',
                 'all',
                 'any',
                 'both',
                 'each',
                 'few',
                 'more',
                 'most',
                 'other',
                 'some',
                 'such',
                 'no',
                 'nor',
                 'not',
                 'only',
                 'own',
                 'same',
                 'so',
                 'than',
                 'too',
                 'very',
                 's',
                 't',
                 'can',
                 'will',
                 'just',
                 'don',
                 "don't",
                 'should',
                 "should've",
                 'now',
                 'd',
                 'll',
                 'm',
                 'o',
                 're',
                 've',
                 'y',
                 'ain',
                 'aren',
                 "aren't",
                 'couldn',
                 "couldn't",
                 'didn',
                 "didn't",
                 'doesn',
                 "doesn't",
                 'hadn',
                 "hadn't",
                 'hasn',
                 "hasn't",
                 'haven',
                 "haven't",
                 'isn',
                 "isn't",
                 'ma',
                 'mightn',
                 "mightn't",
                 'mustn',
                 "mustn't",
                 'needn',
                 "needn't",
                 'shan',
                 "shan't",
                 'shouldn',
                 "shouldn't",
                 'wasn',
                 "wasn't",
                 'weren',
                 "weren't",
                 'won',
                 "won't",
                 'wouldn',
                 "wouldn't"]


def dynamic_loading(path: str) -> pd.DataFrame:
    """
    Dynamically load jsonl, json, or csv.

    :param: path (str) File path

    :return: DataFrame of data loaded
    """
    if not os.path.exists:
        raise Exception(f'Error loading data (filepath "{path}" does not exist)')

    extension = path.split(".")[-1]
    if extension == "jsonl":
        try:
            return pd.read_json(path, lines=True)
        except Exception:
            return pd.read_json(path)
        return pd.read_json(path, lines=True)
    elif extension == "json":
        try:
            return pd.read_json(path)
        except Exception:
            return pd.read_json(path, lines=True)
        return pd.read_json(path)
    elif extension == "csv":
        return pd.read_csv(path)
    else:
        raise Exception("Error loading data (must be json, jsonl, or csv)")


def load_predictions(paths, task="classification"):
    """
    Load Predictions from path(s).

    :param: paths (array or str)

    :return: predictions (dict)
    """
    if type(paths) is str:
        paths = [paths]
    predictions = []
    for path in paths:
        predictions.append(dynamic_loading(path))
    if len(predictions) > 1:
        predictions = pd.concat(predictions)
    else:
        predictions = predictions[0]

    if predictions is not None and task == "classification":
        assert all(c in predictions.columns for c in ["gold", "pred"]), "Data must contain gold and pred fields"
        predictions["correct"] = predictions.apply(lambda x: True if x["gold"] == x["pred"] else False, axis=1)
        predVals = set(predictions["pred"].unique())
        if len(set(predictions.columns).intersection(predVals)) == len(predVals):
            predictions["confidence"] = predictions.apply(lambda x: x[x["pred"]], axis=1)
    predictions["frequency"] = 1
    return predictions


def load_simulations(paths):
    """
    Load simulations from path(s).
    
    :param: paths (array or str)
    
    :return: simulations (dict)
    """

    def make_pred_and_gold(row):
        if row["outcome.True_Negative"] == 1:
            return 0, 0
        elif row["outcome.False_Negative"] == 1:
            return 0, 1
        elif row["outcome.True_Positive"] == 1:
            return 1, 1
        else:
            return 1, 0

    if type(paths) is str:
        paths = [paths]
    simulations = []
    for path in paths:
        sims = dynamic_loading(path)
        sims["dataset"] = "simulation"
        sims["predType"] = "binary"
        if "model_alias" not in sims.columns:
            sims["model_alias"] = path.split("/")[-1].rsplit(".", 1)[0]
        if "pred" not in sims.columns and "gold" not in sims.columns:
            sims[["pred", "gold"]] = sims.apply(make_pred_and_gold, axis=1, result_type='expand')
        simulations.append(sims)
    if len(simulations) > 1:
        simulations = pd.concat(simulations)
    else:
        simulations = simulations[0]
    return simulations


def _find_unique_ngrams_per_class(data_df, goldCol):
    unique_df = data_df.groupby(["Ngrams"], as_index=False)[goldCol].value_counts()
    grams_to_drop = []
    for i, grp in unique_df.groupby("Ngrams"):
        if grp[goldCol].nunique() > 1:
            grams_to_drop.append(i)
    unique_df = unique_df.loc[~unique_df["Ngrams"].isin(grams_to_drop)]
    return unique_df


def _find_common_ngrams(temp_df, predID, verbose=False): 
    set_df = temp_df.groupby(predID)["Ngrams"].apply(set).to_frame().reset_index(drop=False)
    first = set_df.iloc[0]["Ngrams"]

    for i, grp in set_df.groupby(predID):
        first.intersection(grp["Ngrams"].tolist()[0])
        if len(first) == 0:
            if verbose:
                print("No Common Ngrams")
            break
    if len(first) > 0:
        return list(first)
    else:
        return []


def ngrams(
        predictions,
        textCol,
        predID,
        goldCol=None,
        taskCol=None,
        gram_min=2,
        gram_max=3,
        threshold=100,
        topN=20,
        verbose=False,
        returnKeywords=False,
):
    """Find common ngrams."""
    predictions["split"] = predictions[textCol].apply(lambda x: " ".join([word.lower() for word in x.split()]))
    predictions["Ngrams"] = predictions["split"].apply(
        lambda x: " ".join([word for word in x.split() if word not in english_stops])
    )
    predictions["Ngrams"] = predictions["Ngrams"].apply(lambda x: list(everygrams(x.split(), gram_min, gram_max)))
    predictions = predictions.explode("Ngrams")
    predictions["Ngrams"] = predictions["Ngrams"].apply(lambda x: " ".join(x))

    # Remove Ngrams that only occur once
    occur_once = predictions["Ngrams"].value_counts().to_frame()
    occur_once = occur_once.loc[occur_once["Ngrams"] == 1].index
    predictions = predictions.loc[~predictions["Ngrams"].isin(occur_once)]

    # Remove Ngrams found in all documents
    common_ngrams = _find_common_ngrams(predictions, predID)
    if len(common_ngrams) > 0:
        predictions = predictions.loc[~predictions["Ngrams"].isin(common_ngrams)]

    counts = predictions["Ngrams"].value_counts().to_frame()
    keywords = counts.loc[counts["Ngrams"] >= threshold].index.tolist()

    #  new_cols = []
    for key in keywords:
        predictions[key] = predictions["Ngrams"].apply(lambda x: "contains" if key in x else "not contains")

    if verbose:
        fig, axes = plt.subplots(ncols=2)
        h = sns.histplot(counts.loc[counts["Ngrams"] >= threshold], bins=50, ax=axes[0], legend=False)
        h.set_title(f"Ngrams with frequency >={threshold} ({len(keywords)} ngrams)")
        h.set_xlabel("Ngram Frequency")
        h.set_ylabel("Number of Ngrams")

        temp_bar = pd.DataFrame(
            {
                f"Ngrams>={threshold}": len(keywords),
                f"Ngrams<{threshold}": predictions["Ngrams"].nunique() - len(keywords),
            },
            index=["Number of Ngrams"],
        )
        b = temp_bar.plot(kind="bar", stacked=True, ax=axes[1])
        b.set_yscale("log")
        plt.tight_layout()
        plt.show()

        IPython.display.display(IPython.display.HTML("<h3>Top NGrams Unique To Each Class</h3>"))
        binary_df = predictions.loc[predictions[taskCol] == "binary"]
        multi_df = predictions.loc[predictions[taskCol] != "binary"]

        unique_binary_df = _find_unique_ngrams_per_class(binary_df, goldCol)
        unique_multi_df = _find_unique_ngrams_per_class(multi_df, goldCol)

        for cls in unique_binary_df[goldCol].unique():
            print(cls, end="\n\n")
            print(
                unique_binary_df.loc[unique_binary_df[goldCol] == cls][["Ngrams", "count"]]
                .sort_values(by="count", ascending=False)
                .head(topN)
            )
            print("-----" * 20, end="\n\n")

        for cls in unique_multi_df[goldCol].unique():
            print(cls, end="\n\n")
            print(
                unique_multi_df.loc[unique_multi_df[goldCol] == cls][["Ngrams", "count"]]
                .sort_values(by="count", ascending=False)
                .head(topN)
            )
            print("-----" * 20, end="\n\n")

    if returnKeywords:
        return predictions, keywords
    return predictions
