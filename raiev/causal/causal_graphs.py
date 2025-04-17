import networkx as nx

import pygraphviz as pgv
from IPython.display import Image, display

import tempfile


def get_mediators(
    G: nx.Graph,
    treatment,
    outcome,
):
    """
    Identify mediator variables in a causal graph between a treatment node and an outcome node. 

    :param G: (nx.Graph) networkx graph object containing directed causal graph
    :param treatment: (str): name of the treatment node in the graph
    :param outcome: (str): name of the outcome node in the graph
    """
    outcome_ancestors = set(nx.ancestors(G, source=outcome))
    treatment_descendants = set(nx.descendants(G, source=treatment))
    mediators = treatment_descendants.intersection(outcome_ancestors)

    return mediators


def get_colliders(
    G: nx.Graph,
    treatment,
    outcome,
):
    """
    Identify collider variables in a causal graph between a treatment node and an outcome node. 

    :param G: (nx.Graph) networkx graph object containing directed causal graph
    :param treatment: (str): name of the treatment node in the graph
    :param outcome: (str): name of the outcome node in the graph
    """
    outcome_descendants = set(nx.descendants(G, source=outcome))
    treatment_descendants = set(nx.descendants(G, source=treatment))
    possible_colliders = outcome_descendants.intersection(treatment_descendants)

    colliders = []
    for possible_collider in possible_colliders:
        paths = nx.all_simple_paths(G, treatment, possible_collider)
        for path in paths:
            if outcome not in path:
                colliders.append(possible_collider)
                break  # avoid duplicates

    return colliders


def _draw(G, groups_of_nodes=None):
    """
    Perform causal discovery to identify relationships between data factors and model prediction outcomes.

    :param path: (str) optional, default "./". path to write output files that are created by the causal discovery algorithm
    :param ident: (str) identify to use in output files created by the causal discovery algorithm
    :param pd: (float) optional, default 3.0. Penalty discount, a parameter of the BOSS algorithm. Higher values will decrease the complexity of the graph.
    """
    with tempfile.TemporaryFile() as f:
        nx.nx_agraph.write_dot(G, f)
        f.seek(0)
        dot = f.read().decode()
    agraph = pgv.AGraph(dot)
    if groups_of_nodes is not None:
        for nodeset in groups_of_nodes:
            agraph.add_subgraph(nodeset['nodes'], name=nodeset['nodeset'], rank='same')
            
    display(Image(agraph.draw(format='png', prog='dot')))


def drawCausalGraph(graph_df, source='cause', target='effect', weight=None, col_types=None,
                    groups_of_nodes=[{'nodes': ["class1", "class2", "class3", "class4"], 
                                         'nodeset':'prediction types'},
                                        {'nodes': [
                                                "model1-short training",
                                                "model1-long training"
                                            ],
                                         'nodeset':'model1'}
                                    ]):
    """
    Draws the causal graph based an edge list dataframe.

    :param graph_df: (DataFrame) - dataframe with causal edge list
    :param source: (string) - column name of causes
    :param target: (string) - column name of effects
    :param weight: (string) - column name of edge weights
    :param col_types: (string) - defaults to None 
    :param weight: (list) - list of dictionaries containing nodesets ({'nodes':[list of ids], 'nodeset':'name for node set'}) to group together in the causal graph.
    """
    graph_df = graph_df.copy()

    if weight is not None:
        graph_df['penwidth'] = graph_df[weight] * 5.0 / graph_df[weight].max()
    else:
        graph_df['penwidth'] = 3

    graph_df['style'] = 'solid'

    idx = graph_df['cause'].apply(lambda x: x.split('-')[0]) == graph_df['effect'].apply(lambda x: x.split('-')[0])
    graph_df.loc[idx, 'style'] = 'invis'

    nodes = list(set(list(graph_df[source].values) + list(graph_df[target].values)))

    if col_types is None:
        col_types = {
            'Outcomes': [n for n in nodes if 'outcome' in n.lower()],
            'Factors': [n for n in nodes if 'factor' in n.lower()],
        }

    color_map = {}
    for n in nodes:
        if n in col_types['Outcomes']:
            color_map[n.replace('Outcome.', '')] = '#377eb8'
        elif n in col_types['Factors']:
            color_map[n.replace('Factor.', '')] = '#4daf4a'
        else:
            color_map[n] = '#e41a1c'

    for col in [source, target]:
        graph_df[col] = graph_df[col].apply(lambda x: x.replace('Outcome.', ''))
        graph_df[col] = graph_df[col].apply(lambda x: x.replace('Factor.', ''))

    G = nx.from_pandas_edgelist(
        graph_df, source=source, target=target, edge_attr=['penwidth', 'style'], create_using=nx.DiGraph()
    )

    for n, node in enumerate(G.nodes):
        G.nodes[node]['color'] = color_map[node]

    _draw(G, groups_of_nodes)
