'''
Log for code executing, including:
1) classes: Timers,Info,Log
'''

from collections import OrderedDict
import numpy as np
import time
import sys

__all__=['Timers','Info','Log']

class Timers(OrderedDict):
    '''
    Timers for code executing.
    For each of its (key,value) pairs,
        key: string
            A name of the timers.
        value: dict
            The variables to help with the functions of the timer.
    Attribues:
        str_form: 's','c'
            When 's', only the last record of each timer will be included in str;
            When 'c', the cumulative time will also be included in str.
    '''
    ALL=0

    def __init__(self,paras=[],str_form='s'):
        '''
        Constructor.
        Parameters:
            paras: list of string.
                The names of the timers.
        '''
        super(Timers,self).__init__()
        if all(isinstance(para,list) for para in paras):
            for para in paras:
                key,value=para
                self[key]=value
        else:
            self.add('Total',*paras)
        self.str_form=str_form

    def __str__(self):
        '''
        Convert an instance to string.
        '''
        lens=[max(14,len(str(key))+2) for key in self]
        result=[]
        result.append((sum(lens)+12+len(self))*'~')
        result.append('Time (s)'.center(12)+'|'+'|'.join([str(key).center(length) for key,length in zip(self,lens)]))
        result.append((sum(lens)+12+len(self))*'-')
        result.append('Task'.center(12)+'|'+'|'.join([('%e'%(self[key]['records'][-1])).center(length) for key,length in zip(self,lens)]))
        if self.str_form=='c':
            result.append('Cumulative'.center(12)+'|'+'|'.join([('%e'%(self.time(key))).center(length) for key,length in zip(self,lens)]))
        result.append(result[0])
        return '\n'.join(result)

    def add(self,*keys):
        '''
        Add timers.
        Parameters:
            keys: list of string
                The names of the timers to be added.
        '''
        for key in keys:
            self[key]={'time':None,'begin':None,'end':None,'last':None,'records':[]}

    def start(self,*keys):
        '''
        Start the timers.
        Parameters:
            keys: list of string
                The timers to be started.
        '''
        if len(keys)==0:keys=['Total']
        if len(keys)==1 and keys[0]==Timers.ALL: keys=self.keys()
        for timer in map(self.get,keys):
            timer['time']=0.0
            timer['last']=0.0
            timer['begin']=time.time()

    def suspend(self,*keys):
        '''
        Suspend the timers.
        Parameters:
            keys: list of string
                The timers to be suspended.
        '''
        if len(keys)==0:keys=['Total']
        if len(keys)==1 and keys[0]==Timers.ALL: keys=self.keys()
        for timer in map(self.get,keys):
            assert timer['time'] is not None
            if timer['end'] is None:
                timer['end']=time.time()
                timer['time']+=timer['end']-timer['begin']
                timer['begin']=None

    def proceed(self,*keys):
        '''
        Continue the timers.
        Parameters:
            keys: list of string
                The timers to be continued.
        '''
        if len(keys)==0:keys=['Total']
        if len(keys)==1 and keys[0]==Timers.ALL: keys=self.keys()
        for key in keys:
            timer=self[key]
            if timer['time'] is None: 
                self.start(key)
            if timer['begin'] is None:
                timer['begin']=time.time()
                timer['end']=None

    def stop(self,*keys):
        '''
        Stop the timers.
        Parameters:
            keys: list of string
                The timers to be stopped.
        '''
        self.record(*keys)
        self.suspend(*keys)

    def reset(self,*keys):
        '''
        Reset the timers.
        Parameters:
            keys: list of string
                The timers to be reset.
        '''
        if len(keys)==0:keys=['Total']
        if len(keys)==1 and keys[0]==Timers.ALL: keys=self.keys()
        for key in keys:
            self[key]={'time':None,'begin':None,'end':None,'last':None,'records':[]}

    def record(self,*keys):
        '''
        Record the timers.
        Parameters:
            keys: list of string
                The timers to be recorded.
        '''
        if len(keys)==0:keys=['Total']
        if len(keys)==1 and keys[0]==Timers.ALL: keys=self.keys()
        for key in keys:
            timer=self[key]
            timer['records'].append(self.time(key)-timer['last'])
            timer['last']=self.time(key)

    def time(self,key):
        '''
        The cumulative time of the timer.
        Parameters:
            key: string
                The timer whose cumulative time is queried.
        '''
        assert self[key]['time'] is not None
        if self[key]['end'] is None:
            return self[key]['time']+time.time()-self[key]['begin']
        else:
            return self[key]['time']

class Info(OrderedDict):
    '''
    Information for code executing.
    For each of its (key,value):
        key: string
            An entry of the information.
        value: object
            The content of the corresponding entry.
    '''

    def __init__(self,paras=[]):
        '''
        Constructor.
        Parameters:
            paras: list of string
                The entries of the information.
        '''
        super(Info,self).__init__()
        for para in paras:
            if isinstance(para,list):
                key,value=para
                self[key]=value
            else:
                self[para]=None

    def __str__(self):
        '''
        Convert an instance to string.
        '''
        lens=[max(len(str(entry)),len(str(content)))+2 for entry,content in self.iteritems()]
        result=[None,None,None,None,None]
        result[0]=(sum(lens)+9+len(self))*'~'
        result[1]='Entry'.center(9)+'|'+'|'.join([str(key).center(length) for key,length in zip(self,lens)])
        result[2]=(sum(lens)+9+len(self))*'-'
        result[3]='Content'.center(9)+'|'+'|'.join([str(content).center(length) for content,length in zip(self.values(),lens)])
        result[4]=result[0]
        temp=[str(content) for content in self.values()]
        return '\n'.join(result)

class Log(object):
    '''
    The log for code executing.
    Attribues:
        name: string
            The name of the log file.
            NOTE: when the log file is the stdout, this attribute is set to be None.
        mode: 'w','w+','a+'
            The mode of the log file.
        timers: dict of Timers
            The timers for the code executing.
        info: dict of Info
            The info of the code executing.
        fout: file
            The log file.
    '''

    def __init__(self,name=None,mode='a+',timers=None,info=None):
        '''
        Constructor.
        Parameters:
            name: string
                The name of the log file.
                NOTE: when the log file is the stdout, this attribute is set to be None.
            mode: 'w','w+','a+'
                The mode of the log file.
            timers: dict of Timers
                The timers for the code executing.
            info: dict of Info
                The info of the code executing.
        '''
        self.name=name
        self.mode=mode
        self.timers={} if timers is None else timers
        self.info={} if info is None else info
        self.fout=None

    def reset(self,name=None,mode=None):
        '''
        Reset the log file.
        '''
        self.close()
        self.name=name
        self.mode=self.mode if mode is None else mode

    def open(self):
        '''
        Open the log file.
        '''
        if self.fout is None:
            if self.name is None:
                self.fout=sys.stdout
            else:
                self.fout=open(self.name,self.mode)

    def close(self):
        '''
        Close the log file.
        '''
        if self.fout not in (None,sys.stdout):
            self.fout.close()
        self.fout=None

    def __lshift__(self,info):
        '''
        Write info to self.fout.
        Parameters:
            info: string
                The information to be written.
        '''
        self.open()
        self.fout.write(str(info))
        self.fout.flush()
        return self
