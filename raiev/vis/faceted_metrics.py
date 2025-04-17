import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import f1_score

def faceted_bars(predictions, bin, model_col='model_alias', correct_col='correct', palette=None, annot=True):
    models = list(predictions[model_col].unique())
    for model_alias in models:
        curr = predictions[predictions[model_col] == model_alias].groupby(bin)[correct_col].apply(
            lambda x: f1_score(list(x), [True]*len(list(x)))).reset_index().rename(columns={correct_col:'F1'}) 

        order = curr[bin].unique()
        
        fig, ax = plt.subplots(1, 1, figsize=(len(order)*1.6, 3))
        ax.spines.top.set(visible=False)
        ax.spines.right.set(visible=False)
        sns.barplot(curr, x=bin, y='F1', order=order, hue=bin, palette=palette, width=0.8, legend=True, ax=ax)

        if annot:
            for container in ax.containers:
                ax.bar_label(container, fmt='%.2f', label_type='edge')
        
        ax.set_xlabel(bin.title())
        ax.set_title(f'F1 by {bin.title()} - {model_alias}')
        ax.legend(bbox_to_anchor=[1,1], title=bin.title())
        
        if annot:
            x1,x2,y1,y2 = plt.axis()
            plt.axis((x1,x2,y1,y2 + 0.1))
            
        plt.subplots_adjust(right=0.7)

def faceted_line(predictions, bin, model_col='model_alias', correct_col='correct', palette=None):
    fig, ax = plt.subplots(figsize=(10,5))
    data = predictions.sort_values(by=bin)[[bin, correct_col, model_col]].groupby([model_col, bin])[correct_col].apply(
        lambda x: f1_score(list(x), [True]*len(list(x)))).reset_index().rename(columns={correct_col:'F1'})
    
    sns.lineplot(data, x=bin, y='F1', hue=model_col, palette=palette, marker='X', ax=ax)
    ax.set_title(f'F1 by {bin} per {model_col}')
    ax.legend(bbox_to_anchor=[1,1], title=model_col)
    ax.set_ylabel('F1')
    ax.set_xlabel(bin)
    ax.set_xticks(list(data[bin].unique()), minor=False)

def faceted_scatter(predictions, bin, model_col='model_alias', correct_col='correct', palette=None):
    fig, ax = plt.subplots(figsize=(10,5))
    data = predictions.sort_values(by=bin)[[bin, correct_col, model_col]].groupby([model_col, bin])[correct_col].apply(
        lambda x: f1_score(list(x), [True]*len(list(x)))).reset_index().rename(columns={correct_col:'F1'})
    
    sns.scatterplot(data, x=bin, y='F1', hue=model_col, palette=palette, alpha=0.5, ax=ax)
    ax.set_title(f'F1 by {bin} per {model_col}')
    ax.legend(bbox_to_anchor=[1,1], title=model_col)
    ax.set_ylabel('F1')
    ax.set_xlabel(bin)

def faceted_stacked(predictions, bin, model_col='model_alias', correct_col='correct'):
    predictions['Correct'] = predictions[correct_col].apply(lambda x: 'Correct' if x else 'Incorrect')
    models = list(predictions[model_col].unique())

    figsize = (10, 5) if len(models) == 1 else (5*len(models), 6)
    
    fig, ax = plt.subplots(1, len(models), sharey=True, squeeze=False, figsize=figsize)
    for i, model_alias in enumerate(models):
        predictions[predictions[model_col] == model_alias].pivot_table(index=bin, columns='Correct', values=correct_col, aggfunc='count').plot.bar(stacked=True, title=model_alias, legend=False, ax=ax[0][i])
        ax[0][i].set_ylabel('Count')
    ax[0][-1].legend(loc='upper left', title='')
    plt.xticks(rotation=0)
    fig.suptitle(f'Correctness Count by {bin}')