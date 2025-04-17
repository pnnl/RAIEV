from tqdm import tqdm
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def make_embedding_plot(edge_df, node_df, edge_dict, node_dict, *, time_col='time', edge_col='Outcome', node_col='Node Type', 
                        edge_dim_col='Edge Coordinates', node_dim_col='Coordinates', edge_text_col='text', node_text_col='text', 
                        time_steps=None, limit=None):
    """
    edge_df (dataframe) : edge data containing time_col, edge_col, edge_text_col
    node_df (dataframe) : node data containing time_col, node_col, node_text_col
    edge_dict (dict) : dictionary containing style for each edge type
    node_dict (dict) : dictionary containing style for each node type
    time_col (str) : column in edge_df and node_df containing times for slider
    edge_col (str) : column in edge_df mapping edge_df to edge_dict
    node_col (str) : column in node_df mapping node_df to node_dict
    edge_dim (lstr) : column in edge_df containing typles of the x and y coordinates for each end of edge
    node_dim (str) : column in node_df containing tuples of x and y coordinates
    edge_text_col (str) : column containing edge text for legend
    node_text_col (str) : column containing node text for legend and hover
    limit (int) : optional, for plot testing on a time subset
    """
    # Remove outcomes that do not appear in edge_dict
    edge_df = edge_df[edge_df[edge_col].apply(lambda x: x in edge_dict.keys())]

    # Extract time steps
    available_time_steps = edge_df[time_col].unique()
    if time_steps is None: 
        time_steps = available_time_steps
        time_steps = time_steps[:limit] if limit is not None else time_steps
    else:
        time_steps = [t for t in time_steps if t in available_time_steps]

    # Build dynamic edges
    edge_traces, edge_lens = {}, [] 
    node_traces, node_lens = {}, []
    for time_step in tqdm(time_steps):
        edges = []
        visible = 'legendonly' if time_step == time_steps[0] else False
        grp = edge_df[edge_df[time_col] == time_step].reset_index(drop=True)
        grp = grp.sort_values(by=edge_text_col)
        for i, row in grp.iterrows():
            edges.append(go.Scatter(x=row[edge_dim_col][0],
                               y=row[edge_dim_col][1],
                               name=row[edge_text_col],
                               mode='lines',
                               opacity=1,
                               line=edge_dict[row[edge_col]],
                               visible=visible))
        edge_traces[time_step] = edges
        edge_lens.append(len(edges))

        nodes = []
        visible = True if time_step == time_steps[0] else False
        for node_type in node_dict.keys():
            grp = node_df[(node_df[node_col] == node_type) & (node_df[time_col] == time_step)].reset_index(drop=True)
            nodes.append(go.Scatter(
                x=grp[node_dim_col].apply(lambda x: x[0]),
                y=grp[node_dim_col].apply(lambda x: x[1]),
                text=grp[node_text_col],
                name=node_type,
                mode='markers',
                marker=node_dict[node_type],
                visible=visible,
                legendrank=1,
                showlegend=True))
        node_traces[time_step] = nodes
        node_lens.append(len(nodes))
    
    # Add edge traces to figure, connect traces to slider
    fig = make_subplots()
    slider_steps = []
    for i, time_step in enumerate(time_steps):
        for trace in edge_traces[time_step]:
            fig.add_trace(trace)
        for trace in node_traces[time_step]:
            fig.add_trace(trace)

        mask = []
        for j, (edge_length, node_length) in enumerate(zip(edge_lens, node_lens)):
            if j == i:
                mask += ['legendonly'] * edge_length
                mask += [True] * node_length
            else:
                mask += [False] * (edge_length + node_length)
    
        slider_steps.append({'method': 'restyle',
                             'args': [{'visible': mask}],
                             'label': time_step})

    # Figure layout parameters
    fig.update_traces(hovertemplate="%{text}<br>(%{x}, %{y}) <extra></extra>")
    fig['layout'].update(minreducedwidth=1000,
                         height=600, width=1000,
                         xaxis_title="", yaxis_title="",
                         plot_bgcolor='white',
                         legend=dict(y=0.9),
                         sliders=[dict(steps=slider_steps, pad=dict(r=0, t=30))],
                         margin=dict(b=50, t=10, l=100, r=100))
    
    xmin, xmax = node_df[node_dim_col].apply(lambda x: x[0]).min()-0.5, node_df[node_dim_col].apply(lambda x: x[0]).max()+0.9
    ymin, ymax = node_df[node_dim_col].apply(lambda x: x[1]).min()-0.5, node_df[node_dim_col].apply(lambda x: x[1]).max()+0.5
    fig.update_xaxes(range=[xmin, xmax])
    fig.update_yaxes(range=[ymin, ymax])
    fig.show()