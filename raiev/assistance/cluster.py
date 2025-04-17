import umap
import numpy as np
import pandas as pd
import seaborn as sns
from llama_cpp import Llama
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.manifold import TSNE, LocallyLinearEmbedding
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis



def projection(data, modelCol, random_seed, reduction_method):
    """
    Dimensionality reduction of data. Methods supported include UMAP, TSNE, PCA, and Locally Linear Embedding (LLE).
    

    :param data: (Pandas DataFrame) Data to project.
    :param modelCol: (str) Column in data to use as index.
    :param random_seed: (float) Set for reproducibility.
    :param reduction_method: (str) one of umap, tsne, pca, lle
    """
    data = data.reset_index(drop=False)
    temp = data.copy()
    temp = temp.set_index(modelCol)
    temp = pd.DataFrame(temp['y'].to_list(), index=temp.index)
    if reduction_method.lower() == "umap":
        reducer = umap.UMAP(random_state=random_seed)
    elif reduction_method.lower() == "tsne":
        numSamples = len(temp)
        reducer = TSNE(n_components=2, perplexity=numSamples-1)
    elif reduction_method.lower() == "pca":
        reducer = PCA(n_components=2, random_state=random_seed)
    elif reduction_method.lower() == "lle":
        reducer = LocallyLinearEmbedding(n_components=2, random_state=random_seed)
    else:
        raise ValueError(f"Reduction method ``{reduction_method}'' not currently supported. Try ``umap'', ``pca'', ``tsne'', or ``lle''.")
    try:
        if len(temp.columns) == 1:
            col = temp.columns[0]
            temp = temp[[col]].groupby(modelCol)[col].apply(lambda x: pd.Series(x.values)).unstack()
        embedding = reducer.fit_transform(temp)
    except ValueError:
        raise ValueError(f"Error with {reduction_method}. "
                         f"Number of features ({temp.shape[1]}) >> number of samples ({temp.shape[0]}). "
                         f"If umap, try tsne. If tsne, try pca.")
    temp["dim0"] = embedding[:, 0]
    temp["dim1"] = embedding[:, 1]
    data["dim0"] = data[modelCol].map(temp["dim0"].to_dict())
    data["dim1"] = data[modelCol].map(temp["dim1"].to_dict())
    return data


def clusterModelsByData(data, modelCol, random_seed, threshold=0.45, numClusters=None, plotInertia=False):
    """
    Use K-Means to cluster instances. Final number of clusters determined using threshold and 'elbow' method
    or by numClusters.

    :param data: (Pandas DataFrame)
    :param modelCol: (str) Column containing the unique id of each model.
    :param random_seed: (float) Set for reproducibility.
    :param threshold: (float) optional, default 0.45.
    :param numClusters: (int) optional, default None. If specified, uses this number of clusters in kmeans.
                            Ignores threshold parameter.
    :param plotInertia: (bool) optional, default False. If True, plot inertia values per cluster from kmeans.
    """
    temp = data.copy()
    temp = temp.reset_index(drop=False)
    numModels = temp[modelCol].nunique()
    temp = temp.set_index(modelCol)
    temp = pd.DataFrame(temp['y'].to_list(), index=temp.index)
    clusters = None
    old_inertia = 100000
    inertias = {}
    if numClusters is None:
        kmeans_obj = None
        # No need to cluster only 1 model
        if numModels == 1:
            clusters = 0
        for i in range(1, numModels):
            # Use kmeans to cluster similarly performing models
            kmeans = KMeans(n_clusters=i, init='random', max_iter=1000, random_state=random_seed)
            y_pred = kmeans.fit_predict(temp)
            # Calculate relative change; if less than threshold, break (elbow)
            inertias[i] = kmeans.inertia_
            if abs((kmeans.inertia_-old_inertia)/old_inertia) > threshold:
                clusters = y_pred
                old_inertia = kmeans.inertia_
                kmeans_obj = kmeans
                if old_inertia == 0:
                    break
            else:
                break
    else:
        kmeans_obj = KMeans(n_clusters=numClusters, init='random', max_iter=1000, random_state=random_seed)
        clusters = kmeans_obj.fit_predict(temp)

    if plotInertia:
        if numClusters is not None and len(inertias) == 0:
            for i in range(1, numClusters+1):
                plotKmeans = KMeans(n_clusters=i, init='random', max_iter=1000, random_state=random_seed)
                _ = plotKmeans.fit_predict(temp)
                inertias[i] = plotKmeans.inertia_
        tmp = pd.DataFrame.from_dict(inertias, orient='index', columns=["Inertia"]).reset_index(drop=False)
        sns.scatterplot(data=tmp, x="index", y="Inertia")
        plt.xlabel("Num Clusters")
        plt.show()
        
    temp["Cluster"] = clusters
    data["Cluster"] = data[modelCol].map(temp["Cluster"].to_dict())
    rankings = rankClusters(temp, kmeans_obj)
    ranks_dict = rankings.set_index("Cluster")["Rank"].to_dict()
    data["Ranked Group"] = data["Cluster"].map(ranks_dict)
    data["WCSS"] = data["Cluster"].map(rankings.set_index("Cluster")["WCSS"].to_dict())
    return data


def rankClusters(df, kmeans_obj):
    """
    Rank clusters based on Within Cluster Sum of Squares (WCSS). Smaller WCSS => higher rank.

    :param df: Pandas DataFrame of data
    :param kmeans_obj: Fitted K-Means object
    """
    numClusters = df["Cluster"].nunique()
    X = df.drop(columns=["Cluster"]).values
    wcss = np.zeros(numClusters)
    # kmeans_obj is only None if clustering didn't happen i.e., only one model
    if kmeans_obj is not None:
        # Recalculate centers since kmeans.cluster_centers_ not guaranteed
        cluster_centers = [X[kmeans_obj.labels_ == i].mean(axis=0) for i in range(numClusters)]
        for point, label in zip(X, kmeans_obj.labels_):
            wcss[label] += np.square(point - cluster_centers[label]).sum()
    rank_df = pd.DataFrame({"WCSS": wcss})
    rank_df = rank_df.sort_values(by="WCSS",
                                  ascending=True).reset_index(drop=False).rename(columns={"index": "Cluster"})
    rank_df = rank_df.reset_index(drop=False).rename(columns={"index": "Rank"})
    return rank_df#.set_index("Cluster")["Rank"].to_dict()


def characterizeClusters_LLM(df, name="highConfidenceErrors", dataCol="y", modelCol="model_alias", 
                             llm=None, model_path=None, n_ctx=2048, n_threads=8):
    """
    Using an llm model, passed using the llm or model_path parameter, generate descriptions of each cluster of models.
    :param df: (Pandas DataFrame) Dataframe with errors per model.
    :param name: (str) Indicates which errors are presented in clusters.
    :param dataCol: (str, default y) Name of column with data.
    :param modelCol: (str, default model_alias). Name of column with unique id.
    :param llm: (llama_cpp model). LLM to use to generate descriptions. (not passed if using model_path parameter)
    :param model_path: (str) path to LLM model .gguf file
    :param n_ctx: (int, default is 2048) The max sequence length to use - note that longer sequence lengths require much more resources
    :param n_threads: (int, default is 8) The number of CPU threads to use, tailor to your system and the resulting performance
    """
    if model_path is not None:
        # Load Llama model once
        llm = Llama(
            model_path=model_path,   
            n_ctx=n_ctx,  # max sequence length
            n_threads=n_threads,  # number of CPU threads to use
            verbose=False
        )

    if name == "highConfidenceErrors":
        df["total error rate"] = df[dataCol].apply(lambda x: round(x[0] / 100.0, 3))
        df["high confidence error rate"] = df[dataCol].apply(lambda x: round(x[1] / 100.0, 3))

        description = f"In this cluster, there are {len(df)} models. High confidence is defined as greater than 0.9. "
        for m_name, high_error in zip(df[modelCol].tolist(), df["high confidence error rate"].tolist()):
            description += f" The model, {m_name}, has an error rate of {high_error} of errors with " \
                   f"high confidence."
        for m_name, total_error in zip(df[modelCol].tolist(), df["total error rate"].tolist()):
            description += f" The model, {m_name}, has an error rate of {total_error} of total errors over " \
                           f"all confidence values."

    else:
        df["high confidence error rate"] = df[dataCol].apply(lambda x: round(x / 100.0, 3))

        description = f"In this cluster, there are {len(df)} models. High confidence is defined as greater than 0.9. "
        for m_name, high_error in zip(df[modelCol].tolist(), df["high confidence error rate"].tolist()):
            description += f" The model, {m_name}, has an error rate of {high_error} of errors with " \
                           f"high confidence."
    task_prompt = "Describe this cluster of models in terms of performance in 3 sentences or less."
    prompt = f"{description} {task_prompt}"
    try:
        output = llm(
            f"[INST] {prompt} [/INST]",  # Prompt
            max_tokens=512,  # Generate up to 512 tokens
        )
        response = output['choices'][0]['text']
    except ValueError:
        # print("ValueError: Prompt input exceeds context window of 2048. Truncating.")
        # Truncate to 2047-{len of task} tokens, then remove any incomplete sentences. Add task to prompt for LLM.
        len_task = len(task_prompt.split(" "))
        trunk_prompt = " ".join(prompt.split(" ")[:2047-len_task]).rsplit(".", 1)[0]
        trunk_prompt = f"{trunk_prompt} {task_prompt}"
        # Try again
        try:
            output = llm(
                f"[INST] {trunk_prompt} [/INST]",  # Prompt
                max_tokens=512,  # Generate up to 512 tokens
            )
            response = output['choices'][0]['text']
        except ValueError:
            response = "Too many models in cluster. Unable to generate description."

    return response


def characterizeClusters(df, name="highConfidenceErrors", dataCol="y", goldCol="gold"):
    """
    Create natural language descriptions about clusters.

    :param df: Pandas DataFrame of data
    :param name: Name of function from called; clues about what values describe.
    :param dataCol: str.
    :param goldCol: str.
    """

    ERROR_RANKINGS_num = [0.0, 0.25, 0.5, 0.75, 1]
    ERROR_RANKINGS_des = ["almost no", "low", "roughly even", "high", "almost all"]

    if name == "highConfidenceErrors":
        df["total error rate"] = df[dataCol].apply(lambda x: x[0]/100.0)
        df["high confidence error rate"] = df[dataCol].apply(lambda x: x[1]/100.0)
        total_desc = df["total error rate"].describe().to_frame().T
        high_desc = df["high confidence error rate"].describe().to_frame().T

        mean_error = total_desc["mean"].values
        max_error = total_desc["max"].values
        min_error = total_desc['min'].values

        mean_high_error = high_desc["mean"].values
        max_high_error = high_desc["max"].values
        min_high_error = high_desc['min'].values

        std_high_error = high_desc['std'].values
        min_idx = np.argmin([abs(min_error - x) for x in ERROR_RANKINGS_num])
        max_idx = np.argmin([abs(max_error - x) for x in ERROR_RANKINGS_num])
        min_high_idx = np.argmin([abs(min_high_error - x) for x in ERROR_RANKINGS_num])
        max_high_idx = np.argmin([abs(max_high_error - x) for x in ERROR_RANKINGS_num])

        if min_idx == max_idx:
            mean_idx = np.argmin([abs(mean_error - x) for x in ERROR_RANKINGS_num])
            mean_bins = ERROR_RANKINGS_des[mean_idx]
            desc = f"Group has on average {mean_bins} total errors (~{int(ERROR_RANKINGS_num[mean_idx]*100)}%),\n"
        else:
            min_bins = ERROR_RANKINGS_des[min_idx]
            max_bins = ERROR_RANKINGS_des[max_idx]
            desc = f"Group has errors ranging from {min_bins} total errors " \
                   f"(~{int(ERROR_RANKINGS_num[min_idx]*100)}%) to {max_bins} total " \
                   f"errors (~{int(ERROR_RANKINGS_num[max_idx]*100)}%),\n"
        if min_high_idx == max_high_idx:
            mean_idx = np.argmin([abs(mean_high_error - x) for x in ERROR_RANKINGS_num])
            mean_high_bins = ERROR_RANKINGS_des[mean_idx]
            desc += f"and on average {mean_high_bins} high confidence errors (~{int(ERROR_RANKINGS_num[mean_idx]*100)}%)."
        else:
            min_idx = np.argmin([abs(min_high_error - x) for x in ERROR_RANKINGS_num])
            max_idx = np.argmin([abs(max_high_error - x) for x in ERROR_RANKINGS_num])
            min_high_bins = ERROR_RANKINGS_des[min_idx]
            max_high_bins = ERROR_RANKINGS_des[max_idx]
            desc += f"</br>and has high confidence errors ranging from {min_high_bins} errors " \
                    f"(~{int(ERROR_RANKINGS_num[min_idx]*100)}%) to {max_high_bins} errors " \
                    f"(~{int(ERROR_RANKINGS_num[max_idx]*100)}%)."

    elif name == "highConfidenceErrorsOnly":
        df["high confidence error rate"] = df[dataCol].apply(lambda x: x / 100.0)
        high_desc = df["high confidence error rate"].describe().to_frame().T
        mean_high_error = high_desc["mean"].values
        max_high_error = high_desc["max"].values
        min_high_error = high_desc['min'].values
        std_high_error = high_desc['std'].values

        min_high_idx = np.argmin([abs(min_high_error - x) for x in ERROR_RANKINGS_num])
        max_high_idx = np.argmin([abs(max_high_error - x) for x in ERROR_RANKINGS_num])

        if min_high_idx == max_high_idx:
            mean_idx = np.argmin([abs(mean_high_error - x) for x in ERROR_RANKINGS_num])
            mean_high_bins = ERROR_RANKINGS_des[mean_idx]
            desc = f"Group has on average {mean_high_bins} high confidence errors relative to total errors " \
                   f"(~{int(ERROR_RANKINGS_num[mean_idx]*100)}%)."
        else:
            min_idx = np.argmin([abs(min_high_error - x) for x in ERROR_RANKINGS_num])
            max_idx = np.argmin([abs(max_high_error - x) for x in ERROR_RANKINGS_num])
            min_high_bins = ERROR_RANKINGS_des[min_idx]
            max_high_bins = ERROR_RANKINGS_des[max_idx]
            desc = f"Group has high confidence errors ranging from {min_high_bins} errors " \
                   f"(~{int(ERROR_RANKINGS_num[min_idx]*100)}%) to {max_high_bins} errors " \
                   f"(~{int(ERROR_RANKINGS_num[max_idx]*100)}%) relative to total errors."
    elif name == "highConfidenceErrorsByClass":
        df["high confidence error"] = df[dataCol].apply(lambda x: x / 100.0)
        temp = df.loc[df[goldCol] != "Overall"].copy()
        temp["bins"] = temp["high confidence error"].apply(
            lambda x: ERROR_RANKINGS_des[np.argmin([abs(x - err) for err in ERROR_RANKINGS_num])])
        desc = "Group has "
        for i, row in temp.iterrows():
            desc += f"{row['bins']} high confident errors from class {row[goldCol]}, "
        desc = desc[:-2] + ".\n"
        temp = temp.sort_values(by="high confidence error", ascending=True).reset_index(drop=True)
        temp["Ratio"] = temp["high confidence error"].div(temp["high confidence error"].shift(1)).fillna(0).round(1)
        for i, row in temp.iterrows():
            ratio = row["Ratio"]
            if ratio >= 2:
                prev_class = temp.iloc[i-1][goldCol]
                curr_class = row[goldCol]
                desc += f" High confidence errors from class {curr_class} are {ratio} times larger " \
                        f"than from class {prev_class}."
    else:
        desc = ""
    return desc
