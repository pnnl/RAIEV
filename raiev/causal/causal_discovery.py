import pandas as pd

from itertools import product

import subprocess
import os


def causalDiscovery(df, jar_path='./', output_path='./', ident='analysis 1', pd=3.0):
    """
    Perform causal discovery to identify relationships between data factors and model prediction outcomes.

    :param jar_path (str), default "./". path to the causal-cmd jar file.
    :param output_path: (str) optional, default "./". path to write output files that are created by the causal discovery algorithm
    :param ident: (str) identify to use in output files created by the causal discovery algorithm
    :param pd: (float) optional, default 3.0. Penalty discount, a parameter of the BOSS algorithm. Higher values will decrease the complexity of the graph.
    """
    df.astype(float).to_csv('temp.csv', index=False)

    cmd = f'java -jar {jar_path}/causal-cmd-1.12.0-jar-with-dependencies.jar --algorithm boss --data-type continuous --dataset temp.csv --delimiter comma --score sem-bic-score --prefix {ident} --out {output_path} --penaltyDiscount {pd}'

    p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
    stdout, stderr = p.communicate()

    os.remove("temp.csv")

    return f'{output_path}/{ident}_out.txt'


def correctDirections(graph_df, outcome_columns=[]):
    """
    Switch DAG edge directions based on known information about which variables are outcomes. Prevents outcomes from causing non-outcomes.

    :param graph_df: (pandas DataFrame) optional, dataframe with "effect" and "cause" columns
    :param outcome_columns: (list of str) optional, default = [], list of columns which are considered outcomes (cannot cause non-outcomes). if none are provided, any variables starting with "Outcome." will be used.
    """
    nodes = list(set(list(graph_df['effect'].values) + list(graph_df['cause'].values)))

    if len(outcome_columns) == 0:
        outcome_columns = [c for c in nodes if c.startswith('Outcome.')]

    other_cols = [c for c in nodes if c not in outcome_columns]

    if len(outcome_columns) > 0:
        subset = graph_df[(graph_df['cause'].isin(outcome_columns)) & (graph_df['effect'].isin(other_cols))]
        rest = graph_df[~((graph_df['cause'].isin(outcome_columns)) & (graph_df['effect'].isin(other_cols)))]

        subset = subset.rename(columns={'cause': 'effect', 'effect': 'cause'})

        graph_df = pd.concat([rest, subset])

    graph_df = graph_df.drop_duplicates()

    return graph_df


def connectCategoricals(graph_df):
    """
    Add causal connections between the multiple one hot encoded variables representating one categorical variable.

    :param graph_df: (pandas DataFrame), data frame containing "cause" and "effect" columns
    """
    nodes = list(set(list(graph_df['cause'].values) + list(graph_df['effect'].values)))
    first_parts = [n.split('-')[0] for n in nodes]
    df = pd.DataFrame({'node': nodes, 'first_part': first_parts})

    new_edges = []
    for g, grp in df.groupby('first_part'):

        if len(grp) > 1:

            nodes = list(grp['node'].values)

            combos = pd.DataFrame(list(product(nodes, nodes)), columns=['cause', 'effect'])
            combos = combos[combos['cause'] != combos['effect']]

            new_edges.append(combos)

    graph_df = pd.concat([graph_df] + new_edges)

    return graph_df


def parseCausalCmd(fn):
    """
    Return an edge list dataframe based on the causal-cmd output file.

    :param fn: (string) - filename with causal-cmd output
    """
    with open(fn, 'r') as f:
        lines = f.readlines()

    start = False
    causes = []
    effects = []
    for line in lines:
        if 'Graph Edges' in line:
            start = True
            continue
        if start:

            if '--' not in line:
                break

            _, line = line.split('.', 1)

            if '-->' in line:
                cause = line.split('-->')[0].strip('\n').strip(' ')
                effect = line.split('-->')[1].strip('\n').strip(' ')

                causes.append(cause)
                effects.append(effect)

            elif '---' in line:
                cause = line.split('---')[0].strip('\n').strip(' ')
                effect = line.split('---')[1].strip('\n').strip(' ')

                causes.append(cause)
                effects.append(effect)

                causes.append(effect)
                effects.append(cause)

    df = pd.DataFrame({'cause': causes, 'effect': effects})

    return df
