"""Utilities for Dash visualization."""
from dash import Dash, html, dcc, Input, Output, State, ctx
from jupyter_dash import JupyterDash
import dash_bootstrap_components as dbc
import plotly.express as px
import dash_daq as daq
from . import stat_data, error_data, add_high_confidence
import pandas as pd
import plotly.graph_objects as go


def load_interactive_plots(predictions_df, dataset, group="Gold", port=8050):
    external_stylesheets = [dbc.themes.BOOTSTRAP] 
    app = JupyterDash(__name__, external_stylesheets=external_stylesheets)
    compareOrder = ( list(predictions_df[group].unique())     )
    colors = ["black", "#ff7f0e", "#1f77b4", "#006400", "black", "#ff7f0e", "#1f77b4", "#006400"]
    colorMapping = {cat: colors[i] for i, cat in enumerate(compareOrder)}

    def get_button_group(predictions_df):
        options = [
            {"label": "Almost no chance", "value": 1},
            {"label": "Very unlikely", "value": 5},
            {"label": "Unlikely", "value": 20},
            {"label": "Roughly even chance", "value": 45},
            {"label": "Likely", "value": 55},
            {"label": "Very likely", "value": 80},
            {"label": "Almost certainly", "value": 95},
        ]
        selected_options = []
        for index in range(len(options)):
            if index > 0 and options[index]["value"] > predictions_df["confidence"].min() * 100:
                selected_options.append(options[index - 1])
        if len(selected_options) < len(options):
            selected_options.append(options[-1])
        return selected_options
 
    button_group = dbc.RadioItems(
        id="radios",
        className="btn-group",
        inputClassName="btn-check",
        labelClassName="btn btn-light",
        labelCheckedClassName="active",
        options=get_button_group(predictions_df),
        value=get_button_group(predictions_df)[0]["value"],
    )

    def get_line_plot(predictions_df, group="Gold", range_limits=None, cdf_pdf_flag="cdf"):
        dist_data = stat_data(
            predictions_df,
            compareCol=group,
            confidenceCol="confidence",
            round_confidence=2 
        )
        dist_data_df = pd.concat([dist_data[key] for key in dist_data.keys()])
        dist_data_df["cdf"] = dist_data_df["cdf"] * 100
        dist_data_df["pdf"] = dist_data_df["pdf"] * 100
        line_plot_fig = px.line(
            dist_data_df,
            x="confidence",
            y=cdf_pdf_flag,
            color=group,
            color_discrete_map=colorMapping,
            render_mode="webg1",
        ) 
        if range_limits == None:
            range_limits = [dist_data_df["confidence"].min(), dist_data_df["confidence"].max()]
        else:
            if range_limits[0] < dist_data_df["confidence"].min():
                range_limits = [dist_data_df["confidence"].min(), dist_data_df["confidence"].max()]
        line_plot_fig.update_layout(
            title="Confidence Distribution",
            title_x=0.5,
            plot_bgcolor="rgba(0,0,0,0)",
            height=300, 
            margin=dict(l=60, r=30, t=40, b=20),
            xaxis={
                "title": "Model Confidence",
                "linecolor": "#636363",
                "visible": True,
                "showticklabels": True,
                "range": range_limits,
                "rangeslider": {"visible": True},  # Add range slider
            },
            yaxis={
                "title": "% Pred with Confidence <= x" if cdf_pdf_flag == "cdf" else "% Pred with Confidence = x",
                "linecolor": "#636363",
            },
            showlegend=False,
            legend={"title": ""},
        )
        return line_plot_fig

    def get_bar_plot(predictions_df, group="Gold"):
        error_df = error_data(
            predictions_df,
            group,
            plot_type="error",
            compareOrder=compareOrder 
        )
        # Rounding off y values
        error_df["y"] = error_df["y"].round(2)
        bar_plot_fig = px.bar(
            error_df,
            x=group,
            y="y",
            color=group,
            color_discrete_map=colorMapping,
            text="y", 
        ) 
        bar_plot_fig.update_traces(
            textposition="outside", hovertemplate="Category: %{x}<br>" + "% Errors: %{y}}<extra></extra>"
        )
        bar_plot_fig.update_layout(
            title="High Confidence Errors",
            title_x=0.5,
            height=300,
            plot_bgcolor="rgba(0,0,0,0)",
            bargap=0.3,
            margin=dict(l=80, r=20, t=40, b=100),
            xaxis={"title": "", "showline": True, "linecolor": "#636363", "visible": True, "showticklabels": False},
            yaxis={"title": "% Errors", "linecolor": "#636363"},
            yaxis_range=[0, error_df["y"].max() * 1.1],
            showlegend=True,
            legend={"title": ""},
        )
        return bar_plot_fig

    app.layout = html.Div(
        [
            html.Br(),
            dbc.Row(
                [
                    dbc.Col(
                        daq.BooleanSwitch(on=True, labelPosition="top", id="cdf_pdf_toggle"),
                        width=1,
                    ),
                    dbc.Col("Cumulative Distribution", width=2),
                ]
            ),
            html.Br(),
            html.Br(),
            dbc.Row(
                [
                    dbc.Col(dcc.Graph(figure=get_line_plot(predictions_df, group), id="line_plot"), width=6),
                    # dbc.Row(dbc.Col(button_group,width=5))
                    dbc.Col(dcc.Graph(figure=go.Figure(), id="bar_plot"), width=6),
                ]
            ),
            html.Br(),
            dbc.Col(button_group, width={"size": 5, "offset": 1}),
            dcc.Store(id="predictions_df_id", data=predictions_df.to_dict(orient="index")),
        ]
    )

    @app.callback(
        Output("bar_plot", "figure"),
        Output("line_plot", "figure"),
        Output("radios", "value"),
        Input("cdf_pdf_toggle", "on"),
        Input("radios", "value"),
        Input("line_plot", "relayoutData"),
        State("predictions_df_id", "data"),
    )
    def update_output(toggle_val, radios_value, relayoutData, predictions_df_json):
        predictions_df = pd.DataFrame.from_dict(predictions_df_json, orient="index")
        cdf_pdf_flag = "cdf" if toggle_val else "pdf"
        trigger_id = ctx.triggered_id
        range_limits = [predictions_df["confidence"].min(), predictions_df["confidence"].max()]
        set_radios_value = radios_value
        if trigger_id:
            if trigger_id == "line_plot" and relayoutData and "xaxis.range" in relayoutData:
                range_limits = relayoutData["xaxis.range"]
                set_radios_value = None
            elif trigger_id == "radios" and radios_value:
                range_limits = [radios_value / 100, predictions_df["confidence"].max()]
                
        predictions_df = add_high_confidence(predictions_df, range_limits[0], range_limits[1])
        return (
            get_bar_plot(predictions_df, group),
            get_line_plot(predictions_df, group, range_limits, cdf_pdf_flag),
            set_radios_value,
        )
 
    app.run_server(mode="inline", port=port, host="0.0.0.0")


def load_ternary_plot(prediction_df, col1, col2, col3, color_col=None, size_col=None, hover_col_list=None, port=8050):
    external_stylesheets = [dbc.themes.BOOTSTRAP] 
    app = JupyterDash(__name__, external_stylesheets=external_stylesheets)

    def draw_plot(prediction_df, col1, col2, col3, color_col, size_col, hover_col_list):
        scatter_plot = px.scatter_ternary(
            prediction_df,
            a=col1,
            b=col2,
            c=col3,
            hover_data=hover_col_list,
            color=color_col,
            size=size_col,
            size_max=15,
        )
        scatter_plot.update_layout( 
            title_x=0.5,
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=60, r=60, t=100, b=50),
            xaxis={"linecolor": "#636363"},
            yaxis={"linecolor": "#636363"},
            legend={"title": color_col.replace("_", " ").title()},
        )
        return scatter_plot

    app.layout = html.Div(
        [
            dbc.Row(
                [
                    dbc.Col(html.Div(["Axis-1", dcc.Dropdown(prediction_df.columns, col1, id="dropdown_col1")])),
                    dbc.Col(html.Div(["Axis-2", dcc.Dropdown(prediction_df.columns, col2, id="dropdown_col2")])),
                    dbc.Col(html.Div(["Axis-3", dcc.Dropdown(prediction_df.columns, col3, id="dropdown_col3")])),
                    dbc.Col(
                        html.Div(
                            [
                                "Color",
                                dcc.Dropdown(
                                    prediction_df.columns, color_col if color_col else None, id="dropdown_col4"
                                ),
                            ]
                        )
                    ),
                    dbc.Col(
                        html.Div(
                            [
                                "Size",
                                dcc.Dropdown(
                                    prediction_df.columns, size_col if size_col else None, id="dropdown_col5"
                                ),
                            ]
                        )
                    ),
                    dbc.Col(
                        html.Div(
                            [
                                "Hover",
                                dcc.Dropdown(
                                    prediction_df.columns,
                                    hover_col_list if hover_col_list else None,
                                    multi=True,
                                    id="dropdown_col6",
                                ),
                            ]
                        )
                    ),
                ]
            ),
            html.Div(id="selection_container"),
            html.Div(
                [
                    dcc.Graph(
                        figure=draw_plot(prediction_df, col1, col2, col3, color_col, size_col, hover_col_list),
                        id="scatter_plot",
                    )
                ]
            ),
        ]
    )

    @app.callback(
        Output("scatter_plot", "figure"),
        Input("dropdown_col1", "value"),
        Input("dropdown_col2", "value"),
        Input("dropdown_col3", "value"),
        Input("dropdown_col4", "value"),
        Input("dropdown_col5", "value"),
        Input("dropdown_col6", "value"), 
    )
    def update_output(col1, col2, col3, col4, col5, col6):
        return draw_plot(prediction_df, col1, col2, col3, col4, col5, col6)
 
    app.run_server(mode="inline", port=port, host="0.0.0.0")


def load_scatter_plot(prediction_df, col1, col2, color_col=None, size_col=None, hover_col_list=None, port=8050):
    external_stylesheets = [dbc.themes.BOOTSTRAP] 
    app = JupyterDash(__name__, external_stylesheets=external_stylesheets)

    def draw_scatter(prediction_df, col1, col2, color_col, size_col, hover_col_list):
        scatter_plot = px.scatter(
            prediction_df, x=col1, y=col2, hover_data=hover_col_list, color=color_col, size=size_col, size_max=15
        )
        scatter_plot.update_layout( 
            title_x=0.5,
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=60, r=60, t=40, b=20),
            xaxis={"linecolor": "#636363"},
            yaxis={"linecolor": "#636363"},
            legend={"title": color_col.replace("_", " ").title()}, 
        )
        return scatter_plot

    app.layout = html.Div(
        [
            dbc.Row(
                [
                    dbc.Col(html.Div(["X-Axis", dcc.Dropdown(prediction_df.columns, col1, id="dropdown_col1")])),
                    dbc.Col(html.Div(["Y-Axis", dcc.Dropdown(prediction_df.columns, col2, id="dropdown_col2")])),
                    dbc.Col(
                        html.Div(
                            [
                                "Color",
                                dcc.Dropdown(
                                    prediction_df.columns, color_col if color_col else None, id="dropdown_col3"
                                ),
                            ]
                        )
                    ),
                    dbc.Col(
                        html.Div(
                            [
                                "Size",
                                dcc.Dropdown(
                                    prediction_df.columns, size_col if size_col else None, id="dropdown_col4"
                                ),
                            ]
                        )
                    ),
                    dbc.Col(
                        html.Div(
                            [
                                "Hover",
                                dcc.Dropdown(
                                    prediction_df.columns,
                                    hover_col_list if hover_col_list else None,
                                    multi=True,
                                    id="dropdown_col5",
                                ),
                            ]
                        )
                    ),
                ]
            ),
            html.Div(id="selection_container"),
            html.Div(
                [
                    dcc.Graph(
                        figure=draw_scatter(prediction_df, col1, col2, color_col, size_col, hover_col_list),
                        id="scatter_plot",
                    )
                ]
            ),
        ]
    )

    @app.callback(
        Output("scatter_plot", "figure"),
        Input("dropdown_col1", "value"),
        Input("dropdown_col2", "value"),
        Input("dropdown_col3", "value"),
        Input("dropdown_col4", "value"),
        Input("dropdown_col5", "value"), 
    )
    def update_output(col1, col2, col3, col4, col5):
        return draw_scatter(prediction_df, col1, col2, col3, col4, col5)

 
    app.run_server(mode="inline", port=port, host="0.0.0.0")
