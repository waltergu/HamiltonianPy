'''
Engine and App, including:
1) classes: Status, Engine, App
'''

__all__=['Status','Engine','App']

from collections import OrderedDict
from numpy import array
from numpy.linalg import norm
from Constant import RZERO
import os
import time

class Status(object):
    '''
    This class provides an engine/app with a stauts.
    Attributes:
        name: any hashable object
            The name of the engine/app.
        info: any object
            Additional information of the engine/app.
        data: OrderedDict
            The data of the engine/app.
            In current version, these are the parameters of the engine/app.
        _const_: OrderedDict
            The constant parameters of the engine/app.
        _alter_: OrderedDict
            The alterable parameters of the engine/app.
    '''

    def __init__(self,name='',info=''):
        '''
        Constructor.
        Parameters:
            name: any hashable object
                The name of the engine/app.
            info: any object
                Additional information of the engine/app.
        '''
        self.name=name
        self.info=info
        self.data=OrderedDict()
        self._const_=OrderedDict()
        self._alter_=OrderedDict()

    def __str__(self):
        '''
        Convert an instance to string.
        '''
        result=[]
        result.append(str(self.name))
        if len(self._const_)>0:result.append('_'.join([str(v) for v in self._const_.values()]))
        if len(self._alter_)>0:result.append('_'.join([str(v) for v in self._alter_.values()]))
        result.append(str(self.info))
        return '_'.join(result)

    def update(self,const=None,alter=None):
        '''
        Update the parameters of the engine/app.
        Parameters:
            const, alter: dict, optional
                The new parameters.
        '''
        if const is not None:
            self._const_.update(const)
            self.data.update(const)
        if alter is not None:
            self._alter_.update(alter)
            self.data.update(alter)

    @property
    def const(self):
        '''
        This method returns a string representation of the status containing only the constant parameters.
        '''
        result=[]
        result.append(str(self.name))
        if len(self._const_)>0:result.append('_'.join([str(v) for v in self._const_.values()]))
        result.append(str(self.info))
        return '_'.join(result)

    @property
    def alter(self):
        '''
        This method returns a string representation of the status containing only the alterable parameters.
        '''
        result=[]
        result.append(str(self.name))
        if len(self._alter_)>0:result.append('_'.join([str(v) for v in self._alter_.values()]))
        result.append(str(self.info))
        return '_'.join(result)

    def __le__(self,other):
        '''
        Overloaded operator(<=).
        If self.data is a subset of other.data, return True. Otherwise False.
        '''
        try:
            for key,value in self.data.iteritems():
                if norm(value-other.data[key])>RZERO:
                    return False
            else:
                return True
        except KeyError:
            return False

    def __ge__(self,other):
        '''
        Overloaded operator(>=).
        If other.data is a subset of self.data, return True. Otherwise False.
        '''
        return other.__le__(self)

class Engine(object):
    '''
    This class is the base class for all Hamiltonian-oriented classes.
    Attributes:
        din: string
            The directory where the program reads data.
        dout: string
            The directory where the program writes data.
        status: Status
            The status of this engine.
            In current version,
                status.name: string
                    The name of the engine.
                status.info: string
                    The name of the class of the engine.
        apps: dict of App
            The apps registered on this engine, including the dependence of the apps.
    '''

    def __new__(cls,*arg,**karg):
        '''
        This method automatically initialize the attributes of an Engine instance.
        '''
        result=object.__new__(cls)
        dirs={'din':'.','dout':'.'}
        for key,value in dirs.items():
            setattr(result,key,karg.get(key,value))
            if not os.path.exists(getattr(result,key)):
                os.makedirs(getattr(result,key))
        result.status=Status(name=karg.get('name',''),info=cls.__name__)
        result.status.update(const=karg.get('parameters',None))
        result.apps={}
        return result

    def update(self,**paras):
        '''
        This method update the engine.
        '''
        if len(paras)>0:
            raise NotImplementedError()

    def register(self,app,run=True,enforce_run=False):
        '''
        This method register a new app on the engine.
        Parameters:
            app: App
                The app to be registered on this engine.
            run: logical, optional
                When it is True, the app will be run immediately after the registration.
                Otherwise not.
            enforce_run: logical, optional
                When it is True, app.run will be called to run the app.
                Otherwise, even when run is True, app.run may not be called if the engine thinks that this app has already been run.
            NOTE: the CRITERION to judge whether app.run should be called when run==True and enforce_run==False:
                  whether either app.status.info or app.status<=self.status is False.
        '''
        for obj in [app]+app.dependence:
            self.apps[obj.status.name]=obj
        if run:
            stime=time.time()
            cmp=app.status<=self.status
            if enforce_run or (not app.status.info) or (not cmp):
                if not cmp:self.update(**app.status._alter_)
                app.run(self,app)
                app.status.info=True
                app.status.update(alter=self.status._alter_)
            etime=time.time()
            print 'App %s(name=%s): time consumed %ss.'%(app.__class__.__name__,app.status.name,etime-stime)

    def rundependence(self,name,enforce_run=False):
        '''
        This method runs the dependence of the app specified by name.
        Parameters:
            name: any hashable object
                The name to specify whose dependence to be run.
            enforce_run: logical, optional
                When it is True, the run attributes of all the dependence, which are functions themsevles, will be called.
        '''
        for app in self.apps[name].dependence:
            cmp=app.status<=self.status
            if enforce_run or (not app.status.info) or (not cmp):
                if not cmp: self.update(**app.status._alter_)
                self.update(**app.status._alter_)
                app.run(self,app)
                app.status.info=True
                app.status.update(alter=self.status._alter_)

class App(object):
    '''
    This class is the base class for those implementing specific tasks based on Engines.
    Attributes:
        status: Status
            The status of the app.
            In current version,
                status.name: any hashable object
                    The id of the app.
                status.info: logical
                    When True, it means the function app.run has been called by the engine it registered on at least once.
                    Otherwise not.
        dependence: list of App
            The apps on which this app depends.
        plot: logical
            A flag to tag whether the results are to be plotted.
        show: logical
            A flag to tag whether the plotted graph is to be shown.
        parallel: logical
            A flag to tag whether the calculating process is to be paralleled.
        np: integer
            The number of processes used in parallel computing and 0 means the available maximum.
        save_data: logical
            A flag to tag whether the generated data of the result is to be saved on the hard drive.
        run: function
            The function called by the engine to carry out the tasks, which should be implemented by the subclasses of Engine.
    '''

    def __new__(cls,*arg,**karg):
        '''
        This method automatically initialize the attributes of an App instance.
        '''
        result=object.__new__(cls)
        result.status=Status(name=karg.get('name',id(result)),info=False)
        result.status.update(alter=karg.get('parameters',None))
        attr_def={'dependence':[],'plot':True,'show':True,'parallel':False,'np':0,'save_data':True,'run':None}
        for key,value in attr_def.items():
            setattr(result,key,karg.get(key,value))
        return result
