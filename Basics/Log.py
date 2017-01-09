'''
Log for code executing, including:
1) classes: Timer,Timers,Info,Log
'''

from collections import OrderedDict
import numpy as np
import time
import sys

__all__=['Timer','Timers','Info','Log']

class Timer(object):
    '''
    Timer.
    Attribues:
        _time_,_begin_,_end_,_last_: np.float64
            The auxiliary variables of the timer.
        records: list of float64
            The accumulative times between records.
    '''

    def __init__(self):
        '''
        Constructor.
        '''
        self._time_=None
        self._begin_=None
        self._end_=None
        self._last_=None
        self.records=[]

    def start(self):
        '''
        Start the timer.
        '''
        self._time_=0.0
        self._last_=0.0
        self._begin_=time.time()

    def suspend(self):
        '''
        suspend the timer.
        '''
        assert self._time_ is not None
        if self._end_ is None:
            self._end_=time.time()
            self._time_+=self._end_-self._begin_
            self._begin_=None

    def proceed(self):
        '''
        Continue the timer.
        '''
        if self._time_ is None:
            self.start()
        if self._begin_ is None:
            self._begin_=time.time()
            self._end_=None

    def reset(self):
        '''
        Reset the timer.
        '''
        self._time_=None
        self._begin_=None
        self._end_=None
        self._last_=None
        self.records=[]

    def record(self):
        '''
        Record the accumulative time of the timer since the last record.
        '''
        self.records.append(self.time-self._last_)
        self._last_=self.time

    @property
    def time(self):
        '''
        Return the accumulative time of the timer.
        '''
        assert self._time_ is not None
        if self._end_ is None:
            return self._time_+time.time()-self._begin_
        else:
            return self._time_

    def __enter__(self):
        '''
        Used to implement the 'with' statement.
        '''
        self.proceed()
        return self

    def __exit__(self,type,value,traceback):
        '''
        Used to implement the 'with' statement.
        '''
        self.suspend()

class Timers(OrderedDict):
    '''
    Timers for code executing.
    For each of its (key,value) pairs,
        key: string
            The name of the timer.
        value: Timer
            The corresponding timer.
    Attribues:
        str_form: 's','c'
            When 's', only the last record of each timer will be included in str;
            When 'c', the cumulative time will also be included in str.
    '''
    ALL=0

    def __init__(self,names=[],str_form='s'):
        '''
        Constructor.
        Parameters:
            names: list of string.
                The names of the timers.
        '''
        super(Timers,self).__init__()
        if all(isinstance(pair,list) for pair in names):
            for pair in names:
                key,value=pair
                self[key]=value
        else:
            self.add('Total',*names)
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
        result.append('Task'.center(12)+'|'+'|'.join([('%e'%(self[key].records[-1])).center(length) for key,length in zip(self,lens)]))
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
            self[key]=Timer()

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
            timer.start()

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
            timer.suspend()

    def proceed(self,*keys):
        '''
        Continue the timers.
        Parameters:
            keys: list of string
                The timers to be continued.
        '''
        if len(keys)==0:keys=['Total']
        if len(keys)==1 and keys[0]==Timers.ALL: keys=self.keys()
        for timer in map(self.get,keys):
            timer.proceed()

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
        for timer in map(self.get,keys):
            timer.reset()

    def record(self,*keys):
        '''
        Record the timers.
        Parameters:
            keys: list of string
                The timers to be recorded.
        '''
        if len(keys)==0:keys=['Total']
        if len(keys)==1 and keys[0]==Timers.ALL: keys=self.keys()
        for timer in map(self.get,keys):
            timer.record()

    def time(self,key):
        '''
        The cumulative time of the timer.
        Parameters:
            key: string
                The timer whose cumulative time is queried.
        '''
        return self[key].time

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
