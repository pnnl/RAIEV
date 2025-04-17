import IPython
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from .utils import extract_ngrams

def _find_unique_ngrams_per_class(data_df, goldCol):
            unique_df = data_df.groupby(["Ngrams"], as_index=False)[goldCol].value_counts()
            grams_to_drop = []
            for i, grp in unique_df.groupby("Ngrams"):
                if grp[goldCol].nunique() > 1:
                    grams_to_drop.append(i)
            unique_df = unique_df.loc[~unique_df["Ngrams"].isin(grams_to_drop)]
            return unique_df
      
    
def _find_common_ngrams(temp_df, predID):
        num_docs = temp_df[predID].nunique()
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
            
def ngrams(predictions, textCol, predID, goldCol=None, taskCol=None, gram_min=2, gram_max=3, 
           threshold=100, topN=20, verbose=False, returnKeywords=False):
        
        ngrams = extract_ngrams(predictions, id_col=predID, text_col=textCol, ngram_col='Ngrams',
                                minN=gram_min, maxN=gram_max)
        ngrams = ngrams.reset_index()
        predictions = predictions.merge(ngrams, on=predID, how='right')
        predictions = predictions.explode('Ngrams').reset_index(drop=True)
        
        # Remove Ngrams that only occur once
        occur_once = predictions["Ngrams"].value_counts().to_frame()
        occur_once = occur_once.loc[occur_once["Ngrams"]==1].index
        predictions = predictions.loc[~predictions["Ngrams"].isin(occur_once)]
        
        # Remove Ngrams found in all documents
        common_ngrams = _find_common_ngrams(predictions, predID)
        if len(common_ngrams) > 0:
            predictions = predictions.loc[~predictions["Ngrams"].isin(common_ngrams)]
        
        counts = predictions['Ngrams'].value_counts().to_frame()
        keywords = counts.loc[counts['Ngrams']>=threshold].index.tolist()
        
        new_cols = []
        for key in keywords:
            predictions[key] = predictions["Ngrams"].apply(lambda x: "contains" if key in x else "not contains")
            
        if verbose:
            fig, axes = plt.subplots(ncols=2)
            h = sns.histplot(counts.loc[counts['Ngrams']>=threshold], bins=50, ax=axes[0], legend=False)
            h.set_title(f"Ngrams with frequency >={threshold} ({len(keywords)} ngrams)")
            h.set_xlabel("Ngram Frequency")
            h.set_ylabel("Number of Ngrams")

            temp_bar = pd.DataFrame({f"Ngrams>={threshold}": len(keywords), 
                                     f"Ngrams<{threshold}": predictions["Ngrams"].nunique()-len(keywords)}, index=["Number of Ngrams"])
            b = temp_bar.plot(kind='bar', stacked=True, ax=axes[1])
            b.set_yscale("log")
            plt.tight_layout()
            plt.show()
        
            IPython.display.display(IPython.display.HTML('<h3>Top NGrams Unique To Each Class</h3>'))
            binary_df = predictions.loc[predictions[taskCol]=="binary"]
            multi_df = predictions.loc[predictions[taskCol]!="binary"]
            
            unique_binary_df = _find_unique_ngrams_per_class(binary_df, goldCol)
            unique_multi_df = _find_unique_ngrams_per_class(multi_df, goldCol)
            
            for cls in unique_binary_df[goldCol].unique():
                print(cls, end="\n\n")
                print(unique_binary_df.loc[unique_binary_df[goldCol]==cls][["Ngrams","count"]].sort_values(by="count", ascending=False).head(topN))
                print("-----"*20, end="\n\n")
                
            for cls in unique_multi_df[goldCol].unique():
                print(cls, end="\n\n")
                print(unique_multi_df.loc[unique_multi_df[goldCol]==cls][["Ngrams","count"]].sort_values(by="count", ascending=False).head(topN))
                print("-----"*20, end="\n\n")
        
        if returnKeywords:
            return predictions, keywords
        return predictions
