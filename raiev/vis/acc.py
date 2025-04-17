from plotly.subplots import make_subplots
import plotly.graph_objects as go
import pandas as pd
import plotly.express as px
import ipywidgets as widgets
 
from IPython.display import display
import IPython

COLORMAP_OPTIONS = px.colors.named_colorscales()

def drawCCplot(preddf, 
               rowVar = 'pred',
               colVar = 'gold',
               innerVar = 'class',
               includeOverall= False,
               onlyOverall = False,
               overallColor = 'skyblue',
               cmap = 'plasma',
               normalize='table'): 
    
    nrows = preddf[rowVar].nunique()
    ncols = preddf[colVar].nunique()  
    
    
    xLabel = 'Predicted Label' if colVar == 'pred' else colVar
    yLabel = 'Actual Label'  if rowVar == 'gold' else rowVar
    
    if normalize == 'table':
      share_yaxes_value = 'all'
    elif normalize =='cell':
      share_yaxes_value = False
    elif normalize =='columns':
      share_yaxes_value = 'columns'
    elif normalize =='rows':
      share_yaxes_value = 'rows'
      
        
    subplot_bar = make_subplots(rows=nrows, cols=ncols, 
                                specs=[[{"type": "bar"}] * ncols] * nrows,   
                                y_title=yLabel,
                                x_title=xLabel,  
                                shared_yaxes=share_yaxes_value
                               )
    subplot_bar.update_layout(margin=dict(t=0))

    colormapOptions = [x[1] for x in px.colors.get_colorscale(cmap)]
    innerVarOptions = sorted(preddf[innerVar].unique())
    colors = {iv:c for iv,c in zip(innerVarOptions,colormapOptions)}
    colors.update({'Overall':overallColor})
        
    rowIDX = 1
    for row,rdf in preddf.groupby(rowVar):
        colIDX = 1
        for col,cdf in rdf.groupby(colVar):
            bars = cdf.assign(n=1).groupby(innerVar,as_index=False)[['n']].sum()
            if onlyOverall:
                bars = pd.DataFrame([{'n':bars['n'].sum(),
                                                    innerVar:'Overall'}])
            if includeOverall:
                bars = pd.concat([bars,pd.DataFrame([{'n':bars['n'].sum(),
                                                    innerVar:'Overall'}])])

            subplot_bar.add_trace(go.Bar(y=bars['n'], x=bars[innerVar], 
                                         marker_color=bars[innerVar].map(colors),
                                         name=""
                                        ), 
                                  row=rowIDX, col=colIDX)  
            subplot_bar.update_xaxes(title_text=f'<b>{col}</b>', row=nrows, col=colIDX) 
            colIDX+=1
            
        subplot_bar.update_yaxes(title_text=f'<b>{row}</b>', row=rowIDX, col=1)    
        rowIDX+=1
 
    subplot_bar.update_layout(showlegend=False)
    
    subplot_bar.for_each_xaxis(lambda x: x.update(showgrid=False))
    subplot_bar.for_each_yaxis(lambda x: x.update(showgrid=False))

    subplot_bar.show()
    
    

def interactiveCCplot(preddf, 
               rowVar = 'gold',
               colVar = 'pred',
               innerVar = 'class',
               includeOverall= False,
               onlyOverall = False,
               overallColor = 'skyblue',
               cmap = 'plasma',
                     hide_rows_and_cols=False,
                     model_col='model_alias',
                     testset_col=None):
    """
    Interactive visual illustrating crosscheck style plots, with distribution across compareAcross parameter inside cells 

    :param preddf: (Pandas DataFrame) containing predictions
    :param rowVar: (str) column name to use for row data splits, default is 'gold'
    :param colVar: (str) column name to use for column data splits, default is 'pred' 
    :param innerVar: (str) column name to compare across within confusion matrix cells 
    :param includeOverall: (boolean) indicating whether to include overall bar in cells, default is False
    :param onlyOverall: (boolean) indicating whether to only show overall bar in cells, default is False
    :param overallColor: (str) color to use for overall comparison bar, default is 'skyblue'
    :param cmap: (str) colormap to color compareAcross bars with, default is 'plasma'
    :param hide_rows_and_cols: (boolean) indicating whether to hide (True) vs. include (False) row and col selection dropdowns, default is False
    """
    
    # default normalization to table
    normalize='table'
    
    args = {
        'rowVar': rowVar, 
        'colVar': colVar, 
        'innerVar': innerVar,
        'includeOverall': includeOverall,
        'onlyOverall': onlyOverall,
        'overallColor': overallColor, 
        'cmap': cmap,
        'normalize':normalize,
        'model_col':model_col,
        'testset_col':testset_col
    }
    

    dd_width = '210px'
    rowDD = widgets.Dropdown(
        layout={'width': dd_width},
        options=preddf.columns, 
        value=args['rowVar'],
        description='Row:',
    )
    colDD = widgets.Dropdown(
        layout={'width': dd_width},
        options=preddf.columns, 
        value=args['colVar'],
        description='Cols:',
    ) 

    innerDD = widgets.Dropdown(
        layout={'width': dd_width},
        options=preddf.columns, 
        value=args['innerVar'],
        description='Compare:',
    ) 
    settingsDD = widgets.Dropdown(
        layout={'width': dd_width},
        options=['Comparison', '+ Overall', 'Overall Only'],
        value='Comparison',
        description='Bars:',
    ) 
    normDD = widgets.Dropdown(
        layout={'width': dd_width},
        options=['table', 'cell'],
        value=args['normalize'], 
        description='Norm:',
    ) 

    def isValueChanged(change):
        if change['type'] == 'change' and change['name'] == 'value': return True
        return False
                      
    def updateCCplot():  
        if args['model_col'] ==  args['innerVar']:
            # comparing across models, 
            if args['testset_col'] == None or preddf[args['testset_col']].nunique() <2:
                # single matrix
                drawCCplot(preddf, rowVar = args['rowVar'], colVar = args['colVar'], innerVar = args['innerVar'],
                   includeOverall= args['includeOverall'],
                   onlyOverall = args['onlyOverall'],
                   overallColor = args['overallColor'], cmap = args['cmap'],
                   normalize=args['normalize']) 
            else:
                for i, (testset, tdf) in enumerate(preddf.groupby(args['testset_col'])):

                    IPython.display.display(IPython.display.HTML(f"<h4>{testset} <i>test set</i></h4>"))
                    drawCCplot(tdf, rowVar = args['rowVar'], colVar = args['colVar'], innerVar = args['innerVar'],
                               includeOverall= args['includeOverall'],
                               onlyOverall = args['onlyOverall'],
                               overallColor = args['overallColor'], cmap = args['cmap'],
                               normalize=args['normalize'])  
                
        else:
            if args['testset_col'] == None or preddf[args['testset_col']].nunique() <2:
                # tabs across models, single test set
                for i, (model, mdf) in enumerate(preddf.groupby(args['model_col'])):
                    IPython.display.display(IPython.display.HTML(f"<h4>{model} <i>model</i></h4>"))
                    drawCCplot(mdf, rowVar = args['rowVar'], colVar = args['colVar'], innerVar = args['innerVar'],
                               includeOverall= args['includeOverall'],
                               onlyOverall = args['onlyOverall'],
                               overallColor = args['overallColor'], cmap = args['cmap'],
                               normalize=args['normalize']) 

            else:
                # implement later
                print('TBImplemented Soon')
                
        

    def rowDD_on_change(change):
        if isValueChanged(change): 
            args['rowVar'] = change['new']
            # update
            with output:
                output.clear_output()

                updateCCplot()

    def colDD_on_change(change):
        if isValueChanged(change): 
            args['colVar'] = change['new']
            # update
            with output:
                output.clear_output()

                updateCCplot()

    def innerDD_on_change(change):
        if isValueChanged(change): 
            args['innerVar'] = change['new']
            # update
            with output:
                output.clear_output()

                updateCCplot()

    def settingsDD_on_change(change):
        if isValueChanged(change):
            args['includeOverall'], args['onlyOverall'] = {'+ Overall': (True, False),
                                           'Overall Only': (False, True),
                                           'Comparison':(False, False)}[change['new']] 
            # update
            with output:
                output.clear_output()

                updateCCplot()

    def normDD_on_change(change):
        if isValueChanged(change):
            args['normalize'] = change['new']
            # update
            with output:
                output.clear_output()

                updateCCplot()

    rowDD.observe(rowDD_on_change)
    colDD.observe(colDD_on_change)
    innerDD.observe(innerDD_on_change)
    settingsDD.observe(settingsDD_on_change)
    normDD.observe(normDD_on_change)

    output = widgets.Output(layout={'height':'600px','overflow_y':'scroll'})  

    if hide_rows_and_cols:
        controls =  widgets.HBox([innerDD,settingsDD,normDD])
    
    else: 
        controls =  widgets.HBox([
                widgets.VBox([ 
                    rowDD,colDD,innerDD ]),
                widgets.VBox([  
                    settingsDD, normDD])
                ])
    
    display(widgets.VBox([controls,output]))
     
    with output:
        updateCCplot()
         
def confusion_matrix_comparisons(preddf, compareAcross, *,
                                 includeOverall= False, onlyOverall = False, 
                                 overallColor = 'skyblue', cmap = 'plasma',
                     model_col='model_alias',
                     testset_col=None):
    """
    Interactive visual illustrating confusion matrix, with distribution across compareAcross parameter inside cells 

    :param preddf: (Pandas DataFrame) containing predictions
    :param compareAcross: (str) column name to compare across within confusion matrix cells 
    :param includeOverall: (boolean) indicating whether to include overall bar in cells, default is False
    :param onlyOverall: (boolean) indicating whether to only show overall bar in cells, default is False
    :param overallColor: (str) color to use for overall comparison bar, default is 'skyblue'
    :param cmap: (str) colormap to color compareAcross bars with, default is 'plasma'
    """
    
    interactiveCCplot(preddf, 
                   rowVar = 'gold',
                   colVar = 'pred',
                   innerVar = compareAcross,
                   includeOverall= includeOverall,
                   onlyOverall = onlyOverall,
                   overallColor = overallColor,
                   cmap = cmap,
                     hide_rows_and_cols=True,
                     #
                     model_col=model_col,
                     testset_col=testset_col)