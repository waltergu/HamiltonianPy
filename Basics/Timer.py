'''
Timer for code executing, including:
1) classes: Timer, TimerLogger
'''

import numpy as np
import time

__all__=['Timer','TimerLogger']

class Timer(object):
    '''
    Timer.
    Attribues:
        _time_,_begin_,_end_,_last_: float64
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

class TimerLogger(object):
    '''
    Timer logger for code executing.
    Attribues:
        keys: list of string
            The jobs to be timed.
    '''

    def __init__(self,*keys):
        '''
        Constructor.
        Parameters:
            keys: list of string.
                The jobs to be timed.
        '''
        self.keys=keys
        for key in keys:
            setattr(self,key,Timer())

    def __str__(self):
        '''
        Convert an instance to string.
        '''
        keys=self.keys
        lens={key:14 if len(key)+2<14 else len(key)+2 for key in keys}
        result=[None,None,None,None,None]
        result[0]=(sum(lens.values())+14)*'-'
        result[1]='{0:s}'.format('Time (seconds)').center(14)+''.join(['{0:s}'.format(key).center(lens[key]) for key in keys])
        result[2]='{0:s}'.format('Current Step').center(14)+''.join(['{0:e}'.format(getattr(self,key).records[-1]).center(lens[key]) for key in keys])
        result[3]='{0:s}'.format('Accumulation').center(14)+''.join(['{0:e}'.format(getattr(self,key).time).center(lens[key]) for key in keys])
        result[4]=result[0]
        return '\n'.join(result)

    def start(self,*keys):
        '''
        Start the timers.
        Parameters:
            keys: list of string
                The timers to be started.
        '''
        if len(keys)==0:
            keys=self.keys
        for key in keys:
            timer=getattr(self,key)
            timer.start()

    def suspend(self,*keys):
        '''
        Suspend the timers.
        Parameters:
            keys: list of string
                The timers to be suspended.
        '''
        if len(keys)==0:
            keys=self.keys
        for key in keys:
            timer=getattr(self,key)
            timer.suspend()

    def proceed(self,*keys):
        '''
        Continue the timers.
        Parameters:
            keys: list of string
                The timers to be continued.
        '''
        if len(keys)==0:
            keys=self.keys
        for key in keys:
            timer=getattr(self,key)
            timer.proceed()

    def reset(self,*keys):
        '''
        Reset the timers.
        Parameters:
            keys: list of string
                The timers to be reset.
        '''
        if len(keys)==0:
            keys=self.keys
        for key in keys:
            timer=getattr(self,key)
            timer.reset()

    def record(self,*keys):
        '''
        Record the timers.
        Parameters:
            keys: list of string
                The timers to be recorded.
        '''
        if len(keys)==0:
            keys=self.keys
        for key in keys:
            timer=getattr(self,key)
            try:
                timer.record()
            except AssertionError:
                raise AssertionError("TimerLogger record error: %s not started yet."%(key))
