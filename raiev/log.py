import os, json
from datetime import datetime

class evallog:
    """
    Logger class for tracking evaluation workflows and analyses
    """

    def __init__(
        self,
        savedir,
        load_last_session=False,
        load_session_name=None
    ):
        self.savedir = savedir
        os.makedirs(self.savedir,exist_ok=True) 
        
        self.logmeta_fn = os.path.join(self.savedir,'logger_metadata.json')
        if os.path.exists(self.logmeta_fn):
            self._log_metadata = self._load_metadata()
            
            if load_last_session: 
                self.current_session = max(self._log_metadata['logs'].keys())
            elif load_session_name is not None:
                try:
                    self.current_session = self._log_metadata['name2id'][load_session_name]
                except:
                    raise Warning("Cannot find session \"{load_session_name}\". Sessions available are:"+'\n'.join(self._log_metadata['name2id'].keys()))
            else:
                self.current_session = int(max(self._log_metadata['logs'].keys()))+1
                self._log_metadata['session_name_history'][self.current_session] = [] 
                self._log_metadata['logs'][self.current_session] = f'log{self.current_session}.json'
            
            self.log_list = self._load_log()
        else:
            self._log_metadata = {
                'logs':{0:'log0.json'},
                'log_start':{0:str(datetime.now())},
                'name2id':{},
                'id2name':{},
                'session_name_history' : {0:[]}
            }
            self.current_session = 0
            self.log_list = []
            
        if self.current_session in self._log_metadata['id2name']:
            self.current_session_name = self._log_metadata['id2name'][self.current_session]
        else:
            self.current_session_name = None

    def _log_fpath(self):
        return os.path.join(self.savedir,self._log_metadata['logs'][self.current_session])
    
    def _load_log(self):
        """
        Load metadata from file
        """ 
        load_fpath = self._log_fpath()
        if os.path.exists(load_fpath):
            return json.load(open(self._log_fpath(), 'r'))
        else:
            return []
    
    def save_log(self):
        """
        Save metadata to file
        """ 
        #tmp = self.load_log()
        json.dump(self.log_list, open(self._log_fpath(),'w'))
        
    def _load_metadata(self):
        """
        Load metadata from file
        """ 
        return json.load(open(self.logmeta_fn,'r'))
    
    def _save_metadata(self):
        """
        Save metadata to file
        """ 
        json.dump(self._log_metadata, open(self.logmeta_fn,'w'))
        
    def save(self):
        self.save_log()
        self._save_metadata()
        
    def log(self, workflow, function, parameters, comment=None,state=None):
        log_item = {
            'time': str(datetime.now()),
            'workflow': workflow,
            'function': function,
            'parameters': parameters
        }
        if comment is not None:
            log_item['comment']=comment
        if state is not None:
            log_item['state']=state
        self.log_list.append(log_item)
        
    def set_session_name(self, new_session_name, sessionID=None, sessionName=None):
        if sessionID is None and sessionName is None:
            self._log_metadata['session_name_history'][self.current_session].append(
                {'date':str(datetime.now()), 'name':new_session_name}
            )
            last_curr_session_name = self.current_session_name
            del self._log_metadata['name2id'][last_curr_session_name]
            
            self.current_session_name = new_session_name
            self._log_metadata['id2name'][self.current_session] = self.current_session_name
            self._log_metadata['name2id'][self.current_session_name] = self.current_session
        elif sessionID is not None:
            if new_session_name in self._log_metadata['name2id'].keys():
                raise Warning('Session Name "{new_session_name}" already in use.')
            
            self._log_metadata['session_name_history'][sessionID].append(
                {'date':str(datetime.now()), 'name':new_session_name}
            )
            
            ##sessionID
            
            last_curr_session_name = self.current_session_name
            del self._log_metadata['name2id'][last_curr_session_name]
            
            self.current_session_name = new_session_name
            self._log_metadata['id2name'][sessionID] = self.current_session_name
            self._log_metadata['name2id'][self.current_session_name] = sessionID
        
        elif sessionName is not None: 
            if sessionName == new_session_name:
                return
            else:
                if sessionName in self._log_metadata['name2id'].keys():
                    sessionID = self._log_metadata['name2id'][sessionName]
        
        
        
        

class notebooklogger:
    def __init__(self, *, cell_header_token='###', include_cell_numbers=False): 
        self.shell = get_ipython()
        self.cell_header_token = cell_header_token
        self.include_cell_numbers = include_cell_numbers
           
    def _history(self, display=False):
        history = self.shell.find_user_code('')
        history = history.replace(self.cell_header_token,f'\n\n{self.cell_header_token}')
        if self.include_cell_numbers:
            cells = history.split(self.cell_header_token)
            historyStr = ''
            for i,cell in enumerate(cells):
                historyStr = historyStr+f'{self.cell_header_token} [{i+1}]\n{self.cell_header_token}'+cell+'\n'
            history = historyStr
        if display:
            print(history)
        else:
            return history
        
    def _populateNewCell(self, contents, cell_header=''):
        contents = f"{self._commentNewCellStart(cell_header)}\n{contents}"
        print(contents)
        payload = dict(
                source='set_next_input',
                text=contents,
                replace=False,
            )
        self.shell.payload_manager.write_payload(payload, single=False)
        
    def _commentNewCellStart(self, cell_header):
        return f'{self.cell_header_token} {cell_header}'
    