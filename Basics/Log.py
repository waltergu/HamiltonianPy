'''
-----------------------------------------------------------
Time statistics, information record and log file management
-----------------------------------------------------------

Log for code executing, including:
    * classes: Timer, Timers, Info, Log
'''

from ..Misc import Tree
import numpy as np
import matplotlib.pyplot as plt
import time
import sys

__all__=['Timer','Timers','Info','Log']

class Timer(object):
    '''
    Timer.

    Attributes
    ----------
    _time_,_begin_,_end_,_last_ : np.float64
        The auxiliary variables of the timer.
    records : list of float64
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

class Timers(Tree):
    '''
    Timers for code executing. For each of its (key,value) pairs,
        * key: string
            The name of the timer.
        * value: Timer
            The corresponding timer.
    '''
    ALL=0

    def __init__(self,*keys):
        '''
        Constructor.

        Parameters
        ----------
        keys : list of string
            The names of the timers.
        '''
        super(Timers,self).__init__(root='Total',data=Timer())
        for key in keys:
            self.add_leaf('Total',key,Timer())
        self['Total'].proceed()

    def __str__(self):
        '''
        Convert an instance to string.
        '''
        return self.tostr()

    def tostr(self,keys=ALL,form='c'):
        '''
        Get the string representation of the timers.

        Parameters
        ----------
        keys : list of string, optional
            The names of the timers to be included in the representation.
        form : 's' or 'c', optional
            * When 's', only the last record of each timer will be included in the representation;
            * When 'c', the cumulative time will also be included in the representation.
        '''
        if keys==Timers.ALL:
            keys=list(self.expand(mode=Tree.DEPTH,return_form=Tree.NODE))
        elif keys is None:
            keys=[self.root]+self.children(self.root)
        lens=[max(12,len(str(key))+2) for key in keys]
        result=[]
        result.append((sum(lens)+12+len(keys))*'~')
        result.append('Time (s)'.center(12)+'|'+'|'.join([str(key).center(length) for key,length in zip(keys,lens)]))
        result.append((sum(lens)+12+len(keys))*'-')
        result.append('Task'.center(12)+'|'+'|'.join([('%.4e'%(self[key].records[-1])).center(length) for key,length in zip(keys,lens)]))
        if form=='c':
            result.append('Cumulative'.center(12)+'|'+'|'.join([('%.4e'%(self.time(key))).center(length) for key,length in zip(keys,lens)]))
        result.append(result[0])
        return '\n'.join(result)

    def graph(self,parents=None,form='c'):
        '''
        Get the pie chart representation of the timers.

        Parameters
        ----------
        parents : list of string, optional
            The names of the parent timers to be converted to a pie chart.
        form : 's' or 'c', optional
            * When 's', only the last record of each timer will be included in the representation;
            * When 'c', the cumulative time will also be included in the representation.
        '''
        def update(piechart,fractions):
            fractions=np.asarray(fractions)
            theta1=0.0
            for fraction,patch,text,autotext in zip(fractions,piechart[0],piechart[1],piechart[2]):
                theta2=theta1+fraction
                patch.set_theta1(theta1*360)
                patch.set_theta2(theta2*360)
                thetam=2*np.pi*0.5*(theta1+theta2)
                xt=1.1*np.cos(thetam)
                yt=1.1*np.sin(thetam)
                text.set_position((xt,yt))
                text.set_horizontalalignment('left' if xt>0 else 'right')
                xt=0.6*np.cos(thetam)
                yt=0.6*np.sin(thetam)
                autotext.set_position((xt,yt))
                autotext.set_text('%1.1f%%'%(fraction*100))
                theta1=theta2
        if parents is None:
            parents=[self.root]
        elif parents==Timers.ALL:
            parents=[parent for parent in self.expand(mode=Tree.WIDTH,return_form=Tree.NODE) if not self.is_leaf(parent)]
        if hasattr(self,'piecharts'):
            for parent in parents:
                fractions=[self[child].records[-1]/self[parent].records[-1] for child in self.children(parent)]
                update(self.piecharts[(parent,'s')],fractions+[1.0-sum(fractions)])
                if form=='c':
                    fractions=[self.time(child)/self.time(parent) for child in self.children(parent)]
                    update(self.piecharts[(parent,'c')],fractions+[1.0-sum(fractions)])
        else:
            self.piecharts={}
            graph=plt.subplots(nrows=len(parents),ncols=2 if form=='c' else 1)
            axes=graph[1].reshape((len(parents),2 if form=='c' else 1)) if isinstance(graph[1],np.ndarray) else np.array([[graph[1]]])
            for i,parent in enumerate(parents):
                axes[i,0].axis('equal')
                axes[i,0].set_title(parent)
                labels=self.children(parent)
                fractions=[self[child].records[-1]/self[parent].records[-1] for child in labels]
                self.piecharts[(parent,'s')]=axes[i,0].pie(fractions+[1.0-sum(fractions)],labels=labels+['others'],autopct='%1.1f%%')
                if form=='c':
                    axes[i,1].axis('equal')
                    axes[i,1].set_title('%s (%s)'%(parent,'Acc.'))
                    fractions=[self.time(child)/self.time(parent) for child in labels]
                    self.piecharts[(parent,'c')]=axes[i,1].pie(fractions+[1.0-sum(fractions)],labels=labels+['others'],autopct='%1.1f%%')
        plt.pause(10**-6)

    def add(self,parent=None,name=None):
        '''
        Add a timer.

        Parameters
        ----------
        parent : string
            The parent timer of the added timer.
        name : string
            The name of the added timer.
        '''
        parent=self.root if parent is None else parent
        self.add_leaf(parent,name,Timer())

    def record(self):
        '''
        Record all the timers.
        '''
        for timer in self.itervalues():
            timer.record()

    def reset(self):
        '''
        Reset all the timers.
        '''
        for timer in self.itervalues():
            timer.reset()

    def time(self,key):
        '''
        The cumulative time of the timer.

        Parameters
        ----------
        key : string
            The timer whose cumulative time is queried.
        '''
        return self[key].time

    def close(self):
        '''
        Close the graph.
        '''
        plt.close()

class Info(object):
    '''
    Information for code executing.

    Attributes
    ----------
    entry : string
        The name for the entries of Info.
    content : string
        The name for the contents of Info.
    entries : list of string
        The entries of the info.
    contents : list of any object
        The contents of the info.
    formats : list of string
        The output formats of the contents.
    '''

    def __init__(self,*entries,**karg):
        '''
        Constructor.

        Parameters
        ----------
        entries : list of string, optional
            The entries of the info.
        '''
        self.entries=entries
        self.entry=karg.get('entry','Entry')
        self.content=karg.get('content','Content')
        self.contents=[None]*len(entries)
        self.formats=['%s']*len(entries)

    def __len__(self):
        '''
        The length of the info.
        '''
        return len(self.entries)

    def __str__(self):
        '''
        Convert an instance to string.
        '''
        N=max(len(self.entry)+2,len(self.content)+2)
        lens=[max(len(str(entry)),len(format%(content,)))+2 for entry,content,format in zip(self.entries,self.contents,self.formats)]
        result=[None,None,None,None,None]
        result[0]=(sum(lens)+N+len(self))*'~'
        result[1]=self.entry.center(N)+'|'+'|'.join([str(key).center(length) for key,length in zip(self.entries,lens)])
        result[2]=(sum(lens)+N+len(self))*'-'
        result[3]=self.content.center(N)+'|'+'|'.join([(format%(content,)).center(length) for content,format,length in zip(self.contents,self.formats,lens)])
        result[4]=result[0]
        return '\n'.join(result)

    def __setitem__(self,entry,content):
        '''
        Set the content of an entry.
        '''
        index=self.entries.index(entry)
        if isinstance(content,tuple):
            assert len(content)==2
            self.contents[index]=content[0]
            self.formats[index]=content[1]
        else:
            self.contents[index]=content

    def __getitem__(self,entry):
        '''
        Get the content of an entry.
        '''
        return self.contents[self.entries.index(entry)]

    @staticmethod
    def from_ordereddict(od,**karg):
        '''
        Convert an OrderedDict to Info.
        '''
        result=Info(*od.keys(),**karg)
        for key,value in od.iteritems():
            result[key]=value
        return result

class Log(object):
    '''
    The log for code executing.

    Attributes
    ----------
    name : string
        The name of the log file.
    mode : 'w','w+','a+'
        The mode of the log file.
    fout : file
        The log file.

    Notes
    -----
    When the log file is the stdout, the attribute `name` is set to be None.
    '''

    def __init__(self,name=None,mode='a+'):
        '''
        Constructor.

        Parameters
        ----------
        name : string
            The name of the log file.
        mode : 'w','w+','a+'
            The mode of the log file.

        Notes
        -----
        When the log file is the stdout, the parameter `name` is set to be None.
        '''
        self.name=name
        self.mode=mode
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

        Parameters
        ----------
        info : string
            The information to be written.
        '''
        self.open()
        self.fout.write(str(info))
        self.fout.flush()
        return self
