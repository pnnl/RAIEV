import plotly.graph_objects as go

def radar(data, labelCol, *, plotCols=None, fill=True, minAxis=0, maxAxis=5, paper_bgcolor="white", 
          plot_bgcolor=None, showlegend=True, gridcolor = "white"):
    """
    Create an interactive (plotly) radar plot. 
    
    
    :param data: (pandas dataframe) data to plot
    :param labelCol: (str) column containing the label (i.e. label specifying each line plotted)
    :param plotCols: (list) list of strings for column names to use when plotting the spokes of the radar (e.g., metric columns that contain the values for each metric that will be plotted along each spoke of radar plot)
    :param fill: (boolean) whether to fill the inside of the plotted line for each label (e.g. each model)
    :param minAxis: (float) minimum value to start the polar axis at (e.g., 0)
    :param maxAxis: (float) maximum value to start the polar axis at (e.g., 1 if plotting metrics between 0 and 1)
    :param paper_bgcolor: (str) color of window behind plot
    :param plot_bgcolor: (str) what color to use for plot backgroun
    :param showlegend: (boolean) whether to show legend or not
    :param gridcolor: (str) color to make the gridlines
    
    """
    if plotCols is None: 
        plotCols = list(set(dat.columns)-set([labelCol]))
         
    fig = go.Figure()

    for i,row in data[list(plotCols)].iterrows():
        
        if fill:
            fig.add_trace(go.Scatterpolar(
                  r=list(row)+[list(row)[0]],
                  theta=plotCols+[plotCols[0]], 
                  fill='toself',  
                  name=labels[i],
                  hoverinfo='text',
            ))
        else:
            fig.add_trace(go.Scatterpolar(
                  r=list(row)+[list(row)[0]],
                  theta=plotCols+[plotCols[0]],
                  name=labels[i],
                  hoverinfo='text'
            ))

    fig.update_layout(
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.1,
            xanchor="center",
            x=0.5
        ),
      polar=dict( 
        radialaxis=dict(
          visible=True,
          range=[minAxis,maxAxis],
            #
            showline = True,
            linewidth = 2,
            gridcolor = gridcolor,
            gridwidth = 2,
        )), 
      showlegend=showlegend, 
    )   
    if paper_bgcolor is not None:
        fig.update_layout(
        paper_bgcolor = paper_bgcolor  
        )
    if plot_bgcolor is not None: fig.update_polars(bgcolor=plot_bgcolor)
    fig.show()
     
    