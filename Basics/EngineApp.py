'''
Engine and App, including:
1) classes: Name, Engine, App
'''

__all__=['Name','Engine','App']

from collections import OrderedDict
from numpy import array
from numpy.linalg import norm
import os
import time

class Name:
    '''
    This class provides an engine with a name.
    Attributes:
        prefix: string
            Description of the engine.
        suffix: string
            Additional remarks of the engine.
        _alter: OrderedDict
            It contains the contant parameters of the engine.
        _const: OrderedDict
            It contains the alterable parameters of the engine.
        _full: OrderedDict
            It contains all the parameters of the engine.
    '''
    
    def __init__(self,prefix='',suffix=''):
        self.prefix=prefix
        self.suffix=suffix
        self._const=OrderedDict()
        self._alter=OrderedDict()
        self._full=OrderedDict()
    
    def __str__(self):
        return self.full

    def update(self,const=None,alter=None):
        if const is not None:
            self._const.update(const)
            self._full.update(const)
        if alter is not None:
            self._alter.update(alter)
            self._full.update(alter)

    @property
    def parameters(self):
        '''
        This method returns a dict containing all the paramters of the engine.
        '''
        result=OrderedDict()
        result.update(self._const)
        result.update(self._alter)
        return result

    @property
    def const(self):
        '''
        This method returns a string containing only contant parameters as the name of the engine.
        '''
        result=self.prefix+'_'
        for obj in self._const.itervalues():
            result+=repr(obj)+'_'
        result+=self.suffix
        return result

    @property
    def alter(self):
        '''
        This method returns a string containing only alterable parameters as the name of the engine.
        '''
        result=self.prefix+'_'
        for obj in self._alter.itervalues():
            result+=repr(obj)+'_'
        result+=self.suffix
        return result

    @property
    def full(self):
        '''
        This method returns a string containing both contant parameters and alterable parameters as the name of the engine.
        '''
        result=self.prefix+'_'
        for obj in self._const.itervalues():
            result+=repr(obj)+'_'
        for obj in self._alter.itervalues():
            result+=repr(obj)+'_'
        result+=self.suffix
        return result

class Engine(object):
    '''
    This class is the base class for all Hamiltonian-oriented classes.
    Attributes:
        din: string
            The directory where the program reads data.
        dout: string
            The directory where the program writes data.
        name: Name
            The name of this engine.
            This attribute is used for the auto-naming of data files to be read or written.
        state: dict
            The current state of the engine.
            It is the parameters of name in this version.
        waiting_list: list
            The names of apps waiting to be run.
            Note not all activated/added apps are in the waiting list.
        apps: dict
            This dict contains all activated apps.
        repertory: dict
            This dict contains all apps added to this engine.
        dependences: dict
            This dict contains the dependence of the apps.
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
        result.name=Name(prefix=karg.get('name',''),suffix=result.__class__.__name__)
        if 'parameters' in karg:
            result.name.update(const=karg['parameters'])
        result.state=result.name._full
        result.waiting_list=[]
        result.apps={}
        result.repertory={}
        result.dependence={}
        return result

    def register(self,app,dependence=[],waiting_list=True):
        '''
        This method register a new app to the engine.
        Parameters:
            app: App
                The app to be added.
            dependence: list of App
                The apps that app depend on.
                These apps will not go into the waiting_list.
            waiting_list: logical, optional
                When it is True, the app will go into the waiting_list. Otherwise not.
        '''
        if waiting_list: self.waiting_list.append(app.id)
        self.repertory[app.id]=app
        self.dependence[app.id]=[]
        for value in dependence:
            if issubclass(value.__class__,App):
                self.repertory[id(value)]=value
                self.dependence[app.id].append(id(value))
            elif value in self.repertory:
                self.dependence[app.id].append(value)
            else:
                raise ValueError('%s addapps error: one of the dependence(%s) not recognized.'%(self.__class__.__name__,value))

    def update(self,**paras):
        '''
        This method update the engine.
        '''
        if len(paras)>0:
            raise NotImplementedError()

    def activate(self,app):
        '''
        This method activates the app.
        Parameters:
            app: App
                The app to be activated.
        '''
        self.apps[app.__class__.__name__]=app

    def verify(self,app):
        '''
        This method judges whether or not an app has already been run according to its state and the engine's state.
        Parameters:
            app: App
                The app to be verified.
        '''
        if hasattr(app,'state'):
            for key,value in app.state.items():
                if norm(array(value)-array(self.state[key]))!=0:
                    return False
            return True
        else:
            return False

    def rundependence(self,id,enforce_run=False):
        '''
        This method runs the dependence of the app specified by id.
        Parameters:
            id: any hashable object, optional
                The id to specify whose dependence to be run.
            enforce_run: logical, optional
                A flag to tell the program whether or not to enforce to run the dependence.
        '''
        apps=[self.repertory[key] for key in self.dependence.get(id,[])]
        for app in apps:
            self.activate(app)
            self.update(**app.paras)
            if not self.verify(app):
                app.run(self,app)
                app.set_state(self.state)

    def runapps(self,id=None,clock=False,enforce_run=False):
        '''
        This method can be used in two different ways:
        1) self.runapps(id=...,clock=...,enforce_run=...)
            In this case, the app specified by id will be run.
        2) self.rundapps(enforce_run=...)
            In this case, the apps specified by those in self.waiting_list will be run.
        Parameters:
            id: any hashable object, optional
                The id to specify the app to be run.
            clock: logical, optional
                A flag to tell the program whether or not to record the time each run app consumed.
                Note in case 2, this parameter is omitted since the time each app in self.waiting_list costs is automatically recorded.
            enforce_run: logical, optional
                A flag to tell the program whether or not to enforce to run the apps.
        '''
        clock=True if id is None else clock
        ids=self.waiting_list if id is None else [id]
        while ids:
            id=ids.pop(0)
            if clock: stime=time.time()
            app=self.repertory[id]
            self.activate(app)
            self.update(**app.paras)
            if enforce_run or (not self.verify(app)):
                app.run(self,app)
                app.set_state(self.state)
            if clock:
                etime=time.time()
                print 'App %s(id=%s): time consumed %ss.'%(app.__class__.__name__,id,etime-stime)

class App(object):
    '''
    This class is the base class for those implementing specific tasks based on Engines.
    Attributes:
        id: any hashable object
            The unique id of the app.
        paras: dict
            The parameters that are transmited to a engine by the app.
        plot: logical
            A flag to tag whether the results are to be plotted.
        show: logical
            A flag to tag whether the plotted graph is to be shown.
        parallel: logical
            A flag to tag whether the calculating process is to be paralleled.
        np: integer
            The number of processes used in parallel computing and 0 means the available maximum.
        save_data: logical
            A flag to tag whether the generated result data is to be saved on the hard drive.
        run: function
            The function called by the engine to carry out the tasks, which should be implemented by the inheriting class of Engine.
        state: dict
            The current state of the app.
            It is the parameters of an engine's name in this version.
    '''

    def __new__(cls,*arg,**karg):
        '''
        This method automatically initialize the attributes of an App instance.
        '''
        result=object.__new__(cls)
        attr_def={'id':id(result),'paras':{},'plot':True,'show':True,'parallel':False,'np':0,'save_data':True,'run':None}
        for key,value in attr_def.items():
            setattr(result,key,karg.get(key,value))
        return result

    def set_state(self,state):
        '''
        Set the state of the app.
        Parameters:
            state: dict
                New state.
        '''
        if not hasattr(self,'state'):self.state={}
        self.state.update(state)
