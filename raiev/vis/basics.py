from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.express as px

def barplot_grid(df,xCol,yCol,rowCol,colCol,cmap='plasma',title=None,subplot_width=300,subplot_height=300,ylims=None): 
    rowVals = df[rowCol].unique() 
    colVals = df[colCol].unique()
    ncols = len(colVals) 
    nrows = len(rowVals)
    subplot_bar = make_subplots(rows=nrows, cols=ncols, 
                                specs=[[{"type": "bar"}] * ncols] * 1,  
                                subplot_titles=colVals,   
                                shared_yaxes=True )
    subplot_bar.update_layout(margin=dict(t=round(subplot_height/10)))  
    subplot_bar.update_xaxes(ticks='outside',showline=True,linecolor='black')
    subplot_bar.update_yaxes(ticks='outside',showline=True,linecolor='black')
    xColVals = list(df[xCol].unique())
    colors = {iv:c for iv,c in zip(xColVals,[x[1] for x in px.colors.get_colorscale(cmap)][:len(xColVals)])} 


    rowIDX = 1
    for row in rowVals:
        rdf = df[df[rowCol]==row]
        colIDX = 1
        for col in colVals:
            #,cdf in rdf.groupby(colVar):
            cdf = rdf[rdf[colCol]==col] 

            subplot_bar.add_trace(go.Bar(y=cdf[yCol], x=cdf[xCol], 
                                         marker_color=cdf[xCol].map(colors),
                                         name=""), 
                                  row=rowIDX, col=colIDX)  
            #subplot_bar.update_xaxes(title_text=f'<b>{col}</b>', row=nrows, col=colIDX) 
            if ylims is not None: subplot_bar.update_yaxes(range=ylims, row=rowIDX, col=colIDX)
            colIDX+=1 

        subplot_bar.update_yaxes(title_text=f'<b>{row}</b>', row=rowIDX, col=1)    
        rowIDX+=1
    if title is not None:
        subplot_bar.update_layout(plot_bgcolor='rgba(0, 0, 0, 0)', showlegend=False, width=subplot_width*ncols,height=subplot_height*nrows)
    else:
        subplot_bar.update_layout(plot_bgcolor='rgba(0, 0, 0, 0)', showlegend=False, width=subplot_width*ncols,height=subplot_height*nrows, title_text=title)
    subplot_bar.for_each_xaxis(lambda x: x.update(showgrid=False))
    subplot_bar.for_each_yaxis(lambda x: x.update(showgrid=False)) 
    subplot_bar.show()