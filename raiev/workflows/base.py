from ..log import evallog
import inspect
import ipywidgets as widgets
from IPython.core.getipython import get_ipython

class RAIEVanalysis:
    def __init__(self, log_savedir, load_last_session=False, load_session_name=None, logger=None,
                 workflow_outline=None, interactive=True): 
        """
        Initialize a Base Analysis class, contains basic logger setup.
        """
        ## logging
        if logger is None:
            if log_savedir is None:
                log_savedir = './logs'
            self.log_savedir = log_savedir
            self.load_last_session = load_last_session
            self.load_session_name = load_session_name
            self.logger = evallog(self.log_savedir, 
                                  load_last_session=self.load_last_session, 
                                  load_session_name=self.load_session_name)
        else:
            self.logger = logger
        self.workflow_outline = workflow_outline
        self.interactive=interactive
        if self.interactive:
            self.notebookSuggestions = notebookSuggestions() 
        
        self.workflow_cells = self._convertWorkflowStr2Cells()
        self.lastSuggested = -1
        

    def saveLog(self):
        """
        Save log of analysis activities
        """
        self.logger.save_log()
        
    def _logFunc(self, WORKFLOW_CATEGORY, params=None, removeParamFields=[],state=None):
        """Log Current Function Run, under given WORKFLOW_CATEGORY"""
        ## log 
        if params is None:
            params = locals() 
        params = cleanParams(params, removeFields=['self']+removeParamFields)
        
        comment = inspect.getdoc(getattr(self, 
                                         inspect.getframeinfo(inspect.currentframe().f_back).function)
                                ).split('\n:')[0].strip()
        
        self.logger.log(WORKFLOW_CATEGORY, # workflow name
                        inspect.stack()[1].function, #function name
                        params, comment=comment,# params and comments
                        state=state) #state tracking
        ##
        
    def _convertWorkflowStr2Cells(self):
        if self.workflow_outline is None:
            return []
        else:
            temp = [x.split('\n') for x in self.workflow_outline.split('##') if x!='']
            for i,row in enumerate(temp): 
                if row[0]==' Instantiation of workflow object': 
                    break
            cells = [[c[0],'\n'.join(c[1:]).strip()] for c in temp[i+1:]] 
            return cells
        
    def suggestWorkflow(self,n=None): 
        """
        Function to suggest workflow to follow. If called from a jupyter notebook where "interactive=True" parameter was set on instantiation, it will populate new cells to run analyses with. Otherwise, it will print out suggested code to run.
        
        :params n: the number of suggested code cells to populate, if a specific number is selected it will populate the next n cells, without repeating on multiple calls of suggestWorkflow.
        """
        self.workflow_cells = self._convertWorkflowStr2Cells() 
        
        if self.interactive:
            if n is None:
                self.notebookSuggestions._populateMultipleCells(self.workflow_cells)
            else:
                self.lastSuggested+=1
                if self.lastSuggested < len(self.workflow_cells)-1:
                    self.notebookSuggestions._populateMultipleCells(self.workflow_cells[self.lastSuggested:self.lastSuggested+n])
                else:
                    self.notebookSuggestions._populateNewCell(contents='## All suggestions in workflow have been recommended.')
        else:
            if n is None:
                for c in self.workflow_cells:
                    print(f'##{c[0]}')
                    print(c[1]) 
            else:
                self.lastSuggested+=1
                if self.lastSuggested < len(self.workflow_cells)-1: 
                    for c in self.workflow_cells[self.lastSuggested:self.lastSuggested+n]:
                        print(f'##{c[0]}')
                        print(c[1]) 
                else:
                    print('## All suggestions in workflow have been recommended.')
                  
        
def cleanParams(params, removeFields=[]):
    for k in removeFields:
        if k in params.keys(): del params[k] 
    # delete copy of logger
    if 'logger' in params.keys(): del params['logger']
    # delete logging params if defaults:
    for k in ['log_savedir','loadLastLog','loadLogByName']:
        if k in params and params[k] in [None, False]:
            del params[k] 
    ##
    return params


class notebookSuggestions:
    def __init__(self, *, cell_header_token='###', include_cell_numbers=False): 
        self.shell = get_ipython()
        self.cell_header_token = cell_header_token
        self.include_cell_numbers = include_cell_numbers
         
    def _commentNewCellStart(self, cell_header):
        return f'{self.cell_header_token} {cell_header}'
    
    def _populateNewCell(self, contents, cell_header=''):
        contents = f"{self._commentNewCellStart(cell_header)}\n{contents}" 
        payload = dict(
                source='set_next_input',
                text=contents,
                replace=False,
            )
        self.shell.payload_manager.write_payload(payload, single=False)
        
    def _populateMultipleCells(self, content_to_populate):
        for i in range(len(content_to_populate)):
            self._populateNewCell('\n'.join(content_to_populate[-i][1:]).strip(), 
                                  cell_header=f'## {content_to_populate[-i][0]}')