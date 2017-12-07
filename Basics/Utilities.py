'''
---------
Utilities
---------

The utilities of the subpackage, including:
    * constants: RZERO
    * classes: Arithmetic, Timer, Timers, Sheet, Log
    * functions: parity, berry_curvature, berry_phase, decimaltostr, ordinal, mpirun
'''

__all__=['RZERO','Arithmetic','Timer','Timers','Sheet','Log','parity','berry_curvature','berry_phase','decimaltostr','ordinal','mpirun']

from copy import copy
from mpi4py import MPI
from ..Misc import Tree
from scipy.linalg import eigh
import numpy as np
import itertools as it
import matplotlib.pyplot as plt
import warnings
import time
import sys

RZERO=10**-10

warnings.filterwarnings("ignore",".*GUI is implemented.*")

class Arithmetic(object):
    '''
    This class defines the base class for those that support arithmetic operations(+,-,*,/,==,!=).
    To realize the full basic arithmetics, the following methods must be overloaded by its subclasses:
        * __iadd__
        * __add__
        * __imul__
        * __mul__
        * __eq__

    Notes
    -----
        * The addition('+') and multiplication('*') operations are assumed to be commutable.
        * The minus sign ('-' in the negative operator and subtraction operator) are interpreted as the multiplication by -1.0
        * The division operation is interpreted as the multiplication by the inverse of the second argument, which should be a scalar.
    '''

    def __pos__(self):
        '''
        Overloaded positive(+) operator.
        '''
        return self

    def __neg__(self):
        '''
        Overloaded negative(-) operator.
        '''
        return self*(-1)

    def __iadd__(self,other):
        '''
        Overloaded self-addition(+=) operator.
        '''
        raise NotImplementedError("%s (+=) error: not implemented."%self.__class__.__name__)

    def __add__(self,other):
        '''
        Overloaded left addition(+) operator.
        '''
        raise NotImplementedError("%s (+) error: not implemented."%self.__class__.__name__)

    def __radd__(self,other):
        '''
        Overloaded right addition(+) operator.
        '''
        return self+other

    def __isub__(self,other):
        '''
        Overloaded self-subtraction(-=) operator.
        '''
        return self.__iadd__(-other)

    def __sub__(self,other):
        '''
        Overloaded subtraction(-) operator.
        '''
        return self+other*(-1.0)

    def __imul__(self,other):
        '''
        Overloaded self-multiplication(*=) operator.
        '''
        raise NotImplementedError("%s (*=) error: not implemented."%self.__class__.__name__)

    def __mul__(self,other):
        '''
        Overloaded left multiplication(*) operator.
        '''
        raise NotImplementedError("%s (*) error: not implemented."%self.__class__.__name__)

    def __rmul__(self,other):
        '''
        Overloaded right multiplication(*) operator.
        '''
        return self*other

    def __idiv__(self,other):
        '''
        Overloaded self-division(/=) operator.
        '''
        return self.__imul__(1.0/other)

    def __div__(self,other):
        '''
        Overloaded left division(/) operator.
        '''
        return self*(1.0/other)

    def __eq__(self,other):
        '''
        Overloaded equivalent(==) operator.
        '''
        raise NotImplementedError("%s (==) error: not implemented."%self.__class__.__name__)

    def __ne__(self,other):
        '''
        Overloaded not-equivalent(!=) operator.
        '''
        return not self==other

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
        * key: str
            The name of the timer.
        * value: Timer
            The corresponding timer.
    '''
    ALL=0

    def __init__(self,*keys,**karg):
        '''
        Constructor.

        Parameters
        ----------
        keys : list of str
            The names of the timers.
        '''
        super(Timers,self).__init__(root=karg.get('root','Total'),data=Timer())
        for key in keys: self.add_leaf(self.root,key,Timer())
        self[self.root].proceed()

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
        keys : list of str, optional
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
        parents : list of str, optional
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
        parent : str
            The parent timer of the added timer.
        name : str
            The name of the added timer.
        '''
        parent=self.root if parent is None else parent
        self.add_leaf(parent,name,Timer())

    def remove(self,timer):
        '''
        Remove a timer and its subtimers.

        Parameters
        ----------
        timer : str
            The timer to be removed.
        '''
        self.remove_subtree(timer)

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
        self[self.root].proceed()

    def time(self,key):
        '''
        The cumulative time of the timer.

        Parameters
        ----------
        key : str
            The timer whose cumulative time is queried.
        '''
        return self[key].time

    @staticmethod
    def close():
        '''
        Close the graph.
        '''
        plt.close()

    def cleancache(self):
        '''
        Clean the cache of the timers.
        '''
        if hasattr(self,'piecharts'): delattr(self,'piecharts')

class Sheet(object):
    '''
    Sheet.

    Attributes
    ----------
    cols,rows : tuple of str
        The tags of the columns/rows of the sheet.
    corner : str
        The tag of the corner of the sheet.
    widths : list of int
        The widths of the columns.
    formats : 2d ndarray of str
        The output formats of the contents.
    contents : 2d ndarray
        The contents of the sheet.
    sheet : 2d ndarray of str
        The str representation of the contents.
    '''

    def __init__(self,cols=(None,),rows=('Content',),corner='Entry',widths=None,formats=None,contents=None):
        '''
        Constructor.

        Parameters
        ----------
        cols,rows : tuple of str, optional
            The tags of the columns/rows of the sheet.
        corner : str, optional
            The tag of the corner of the sheet.
        widths : list of int, optional
            The widths of the columns.
        formats : 2d ndarray, optional
            The output formats of the contents.
        contents : 2d ndarray, optional
            The contents of the sheet.
        '''
        self.cols=tuple(cols)
        self.rows=tuple(rows)
        self.corner=corner
        self.widths=widths
        if widths is not None: assert len(widths)==len(cols)+1
        self.formats=np.array(['%s']*len(rows)*len(cols),dtype=object).reshape((len(rows),len(cols))) if formats is None else formats
        assert self.formats.shape==(len(self.rows),len(self.cols))
        if contents is None:
            self.contents=np.array([None]*len(rows)*len(cols)).reshape((len(rows),len(cols)))
            self.sheet=np.array(['']*len(rows)*len(cols),dtype=object).reshape((len(rows),len(cols)))
        else:
            assert contents.shape==(len(self.rows),len(self.cols))
            self.contents=contents
            self.sheet=np.array([format%content for content,format in zip(self.contents.flatten(),self.formats.flatten())],dtype=object).reshape((len(rows),len(cols)))

    @property
    def shape(self):
        '''
        The shape of the sheet.
        '''
        return self.contents.shape

    def rowindex(self,row):
        '''
        The index of a row.

        Parameters
        ----------
        row : str or int
            The row whose index is inquired.

        Returns
        -------
        int
            The row index.
        '''
        return row if isinstance(row,int) or isinstance(row,long) else self.rows.index(row)

    def colindex(self,col):
        '''
        The index of a column.

        Parameters
        ----------
        col : str or int
            The column whose index is inquired.

        Returns
        -------
        int
            The column index.
        '''
        return col if isinstance(col,int) or isinstance(col,long) else self.cols.index(col)

    def index(self,entry):
        '''
        The index of the input entry.

        Parameters
        ----------
        entry : tuple, str or int
            The input entry.

        Returns
        -------
        2-tuple
            The index of the input entry.
        '''
        try:
            if len(self.rows)==1: return (0,self.colindex(entry))
            if len(self.cols)==1: return (self.rowindex(entry),0)
        except ValueError:
            pass
        assert isinstance(entry,tuple) and len(entry)==2
        return self.rowindex(entry[0]),self.colindex(entry[1])

    def __str__(self):
        '''
        Convert an instance to string.
        '''
        return self.tostr()

    def __setitem__(self,entry,content):
        '''
        Set the content of an entry.
        '''
        index=self.index(entry)
        if isinstance(content,tuple):
            assert len(content)==2
            self.contents[index]=content[0]
            self.formats[index]=content[1]
            self.sheet[index]=content[1]%content[0]
        else:
            self.contents[index]=content
            self.sheet[index]=self.formats[index]%content

    def __getitem__(self,entry):
        '''
        Get the content of an entry.
        '''
        return self.contents[self.index(entry)]

    def tostr(self,rowon=True,colon=True):
        '''
        Convert an instance to string.

        Parameters
        ----------
        rowon,colon : logical, optional
            True for including the tags of the rows/columns in the result and False for not.

        Returns
        -------
        str
            The converted string.
        '''
        sheet,rowadded,coladded=self.sheet,False,False
        if rowon and self.rows!=(None,):
            sheet=np.insert(sheet,0,['{}'.format(tag) for tag in self.rows],axis=1)
            rowadded=True
        if colon and self.cols!=(None,):
            sheet=np.insert(sheet,0,([self.corner] if rowadded else [])+['{}'.format(tag) for tag in self.cols],axis=0)
            coladded=True
        widths=np.max(np.array([len(entry) for entry in sheet.flatten()]).reshape(sheet.shape),axis=0) if self.widths is None else self.widths[(0 if rowadded else 1):]
        result=[None]*(len(sheet)+(3 if coladded else 2))
        length=np.sum(widths)+3*sheet.shape[1]-1
        result[+0]=length*'~'
        result[-1]=length*'~'
        if coladded: result[2]=length*'-'
        for i,row in enumerate(sheet):
            index=i+2 if (coladded and i>0) else i+1
            result[index]='|'.join(entry.center(width+2) for entry,width in zip(row,widths))
            assert len(result[index])==length
        return '\n'.join(result)

    def frame(self,rowon=True):
        '''
        Return the frame of the sheet.
        '''
        assert self.widths is not None
        return '~'*(np.sum(self.widths[(0 if rowon else 1):])+3*(self.sheet.shape[1]+(1 if rowon else 0))-1)

    def division(self,rowon=True):
        '''
        Return the division line of the sheet.
        '''
        assert self.widths is not None
        return '-'*(np.sum(self.widths[(0 if rowon else 1):])+3*(self.sheet.shape[1]+(1 if rowon else 0))-1)

    def coltagstostr(self,corneron=True):
        '''
        Convert the column tags to string.

        Parameters
        ----------
        corneron : logical, optional
            True for including the corner of the sheet and False for not.

        Returns
        -------
        str
            The converted string.
        '''
        assert self.widths is not None
        content=['{}'.format(tag) for tag in self.cols]
        sheet=np.concatenate(([self.corner],content)) if corneron else content
        width=self.widths[(0 if corneron else 1):]
        return '|'.join(entry.center(width+2) for entry,width in zip(sheet,width))

    def rowtagstostr(self,corneron=True):
        '''
        Convert the row tags to string.

        Parameters
        ----------
        corneron : logical, optional
            True for including the corner of the sheet and False for not.

        Returns
        -------
        str
            The converted string.
        '''
        assert self.widths is not None
        return '\n'.join(entry.center(self.widths[0]+2) for entry in it.chain([self.corner] if corneron else [],['{}'.format(tag) for tag in self.rows]))

    def tagtostr(self,tag):
        '''
        Convert the/a/a tag of the corner/rows/columns to string.

        Parameters
        ----------
        tag : str
            The/A/A tag of the corner/rows/columns.

        Returns
        -------
        str
            The converted string.
        '''
        assert self.widths is not None
        if tag==self.corner:
            return self.corner.center(self.widths[0]+2)
        else:
            try:
                index=self.rowindex(tag)
                return '{}'.format(self.rows[index]).center(self.widths[0]+2)
            except ValueError:
                index=self.colindex(tag)
                return '{}'.format(self.cols[index]).center(self.widths[index+1]+2)

    def rowtostr(self,row,rowon=True):
        '''
        Convert a row to string.

        Parameters
        ----------
        row : str or int
            The row to be converted.
        rowon : logical, optional
            True for including the tag of the row.

        Returns
        -------
        str
            The converted string.
        '''
        assert self.widths is not None
        index=self.rowindex(row)
        tag='{}'.format(self.rows[index])
        content=self.sheet[index,:]
        sheet=np.concatenate(([tag],content)) if rowon else content
        width=self.widths[(0 if rowon else 1):]
        return '|'.join(entry.center(width+2) for entry,width in zip(sheet,width))

    def entrytostr(self,entry):
        '''
        Convert an entry of the sheet to string.

        Parameters
        ----------
        entry : tuple, str or int
            The input entry.

        Returns
        -------
        str
            The converted string.
        '''
        assert self.widths is not None
        index=self.index(entry)
        return self.sheet[index].center(self.widths[index[1]+1]+2)

    @staticmethod
    def from_ordereddict(od,mode='C'):
        '''
        Convert an OrderedDict to Sheet.

        Parameters
        ----------
        od : OrderedDict
            The ordered dict that contains the tags and contents of the sheet.
        mode : 'R'/'C', optional
            'R' for a m*1 sheet and 'C' for a 1*m sheet.

        Returns
        -------
        Sheet
            The converted sheet.
        '''
        assert mode in ('R','C')
        result=Sheet(rows=od.keys()) if mode=='R' else Sheet(cols=od.keys())
        for key,value in od.iteritems():
            result[key]=value
        return result

class Log(object):
    '''
    The log for code executing.

    Attributes
    ----------
    dir : str
        The directory where to store the log file.
    name : str
        The name of the log file.
    fout : file
        The log file.

    Notes
    -----
    When the log file is the stdout, the attribute `name` is set to be None.
    '''

    ON=True
    FLUSH=True

    def __init__(self,dir=None,name=None):
        '''
        Constructor.

        Parameters
        ----------
        dir : str, optional
            The directory where to store the log file.
        name : str, optional
            The name of the log file.

        Notes
        -----
        When the log file is the stdout, the parameter `name` is set to be None.
        '''
        self.dir=dir
        self.name=name
        self.fout=None

    def reset(self,name):
        '''
        Reset the log file.
        '''
        self.close()
        self.name=name

    def open(self):
        '''
        Open the log file.
        '''
        if self.fout is None:
            if self.name is None:
                self.fout=sys.stdout
            else:
                self.fout=open('%s/%s'%(self.dir,self.name),'a+')

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
        info : str
            The information to be written.
        '''
        if Log.ON:
            self.open()
            self.fout.write(str(info))
            if Log.FLUSH: self.fout.flush()
        return self

def parity(permutation):
    '''
    Determine the parity of a permutation.

    Parameters
    ----------
    permutation : list of integer
        A permutation of integers from 0 to N-1.

    Returns
    -------
    -1 or +1
        * -1 for odd permutation
        * +1 for even permutation
    '''
    permutation=copy(permutation)
    result=1
    for i in xrange(len(permutation)-1):
        if permutation[i]!=i:
            result*=-1
            pos=min(xrange(i,len(permutation)),key=permutation.__getitem__)
            permutation[i],permutation[pos]=permutation[pos],permutation[i]
    return result

def berry_curvature(H,kx,ky,mu,d=10**-6):
    '''
    Calculate the Berry curvature of the occupied bands for a Hamiltonian with the given chemical potential using the Kubo formula.

    Parameters
    ----------
    H : callable
        Input function which returns the Hamiltonian as a 2D array.
    kx,ky : float
        The two parameters which specify the 2D point at which the Berry curvature is to be calculated.
        They are also the input parameters to be conveyed to the function H.
    mu : float
        The chemical potential.
    d : float, optional
        The spacing to be used to calculate the derivatives.

    Returns
    -------
    float
        The calculated Berry curvature for function H at point kx,ky with chemical potential mu.
    '''
    result=0
    Vx=(H(kx+d,ky)-H(kx-d,ky))/(2*d)
    Vy=(H(kx,ky+d)-H(kx,ky-d))/(2*d)
    Es,Evs=eigh(H(kx,ky))
    for n in xrange(Es.shape[0]):
        for m in xrange(Es.shape[0]):
            if Es[n]<=mu and Es[m]>mu:
                result-=2*(np.vdot(np.dot(Vx,Evs[:,n]),Evs[:,m])*np.vdot(Evs[:,m],np.dot(Vy,Evs[:,n]))/(Es[n]-Es[m])**2).imag
    return result

def berry_phase(H,path,ns):
    '''
    Calculate the Berry phase of some bands of a Hamiltonian along a certain path.

    Parameters
    ----------
    H : callable
        Input function which returns the Hamiltonian as a 2D array.
    path : iterable
        The path along which to calculate the Berry phase.
    ns : iterable of int
        The sequences of bands whose Berry phases are wanted.

    Returns
    -------
    1d ndarray
        The wanted Berry phase of the selected bands.
    '''
    ns=np.array(ns)
    for i,parameters in enumerate(path):
        new=eigh(H(**parameters))[1][:,ns]
        if i==0:
            result=np.ones(len(ns),new.dtype)
            evs=new
        else:
            for j in xrange(len(ns)):
                result[j]*=np.vdot(old[:,j],new[:,j])
        old=new
    else:
        for j in xrange(len(ns)):
            result[j]*=np.vdot(old[:,j],evs[:,j])
    return np.angle(result)/np.pi

def decimaltostr(number,n=5):
    '''
    Convert a number to string.

    Parameters
    ----------
    number : int/long/float/complex
        The number to be converted to string.
    n : int, optional
        The number of decimal fraction to be kept.

    Returns
    -------
    str
        The string representation of the input number.
    '''
    if isinstance(number,int) or isinstance(number,long):
        result=str(number)
    elif isinstance(number,float):
        result=('{:.%sf}'%n).format(number).rstrip('0')
        if result[-1]=='.': result+='0'
    elif isinstance(number,complex):
        real=('{:.%sf}'%n).format(number.real).rstrip('0')
        imag=('{:.%sf}'%n).format(number.imag).rstrip('0')
        temp=[]
        if real!='0.': temp.append('%s%s'%(real,'0' if real[-1]=='.' else ''))
        if imag!='0.': temp.append('%s%s%sj'%('+' if number.imag>0 and len(temp)>0 else '',imag,'0' if imag[-1]=='.' else ''))
        result=''.join(temp)
    else:
        raise TypeError('decimaltostr error: not supported class(%s).'%number.__class__.__name__)
    return result

def ordinal(number):
    '''
    Convert a number to its corresponding ordinal.

    Parameters
    ----------
    number : int
        The number.

    Returns
    -------
    str
        The corresponding ordinal.
    '''
    return '1st' if number==0 else ('2nd' if number==1 else ('3rd' if number==2 else '%sth'%(number+1)))

def mpirun(f,arguments,comm=MPI.COMM_WORLD,bcast=True):
    '''
    Wrapper for the parallel running of f using the mpi4py.

    Parameters
    ----------
    f : callable
        The function to be parallelly run using the mpi4py.
    arguments : list of tuple
        The list of arguments passed to the function f.
    comm : MPI.Comm, optional
        The MPI communicator.
    bcast : True or False
        When True, broadcast the result for all processes;
        Otherwise only the rank 0 process hold the result.

    Returns
    -------
    list
        The returned values of f with respect to the arguments.
    '''
    size=comm.Get_size()
    rank=comm.Get_rank()
    if size>1:
        import mkl
        mkl.set_num_threads(1)
    temp=[]
    for i,argument in enumerate(arguments):
        if i%size==rank:
            temp.append(f(*argument))
    temp=comm.gather(temp,root=0)
    result=[]
    if rank==0:
        for i in xrange(len(arguments)):
            result.append(temp[i%size][i/size])
    if bcast:
        result=comm.bcast(result,root=0)
    return result
