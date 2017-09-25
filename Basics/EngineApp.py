'''
==============
Engine and app
==============

This module defines the general framework to apply algorithms to tasks, including:
    * classes: Engine, App
'''

__all__=['Engine','App']

from DegreeOfFreedom import Status
from Utilities import Timers,Log
from warnings import warn
import os

class Engine(object):
    '''
    This class is the base class for all Hamiltonian-oriented classes.

    Attributes
    ----------
    dlog : string
        The directory where the program records the logs.
    din : string
        The directory where the program reads data.
    dout : string
        The directory where the program writes data.
    status : Status
        The status of this engine.
    preloads : list of App
        The preloaded apps of the engine, which will become the dependences of all the other apps registered on it.
    apps : dict of App
        The apps registered on this engine (the dependences of the apps not included).
    clock : Timers
        The clock of the engine.
    log : Log
        The log of the engine.
    '''
    DEBUG=False
    MKDIR=True

    def __new__(cls,*arg,**karg):
        '''
        This method automatically initialize the attributes of an Engine instance.
        '''
        result=object.__new__(cls)
        dirs={'dlog':'.','din':'.','dout':'.'}
        for key,value in dirs.items():
            setattr(result,key,karg.get(key,value))
            if cls.MKDIR and not os.path.exists(getattr(result,key)):
                os.makedirs(getattr(result,key))
        result.log=Log() if Engine.DEBUG else Log(name=karg.get('log',None),dir=result.dlog,mode='a+')
        result.status=Status(name=karg.get('name',None),info=cls.__name__,data=karg.get('parameters',()),map=karg.get('map',None))
        result.clock=Timers()
        result.preloads=karg.get('preloads',[])
        result.apps={}
        return result

    def update(self,**paras):
        '''
        This method update the engine.
        '''
        if len(paras)>0:
            raise NotImplementedError()

    def register(self,app,run=True):
        '''
        This method register a new app on the engine.

        Parameters
        ----------
        app : App
            The app to be registered on this engine.
        run : logical, optional
            When it is True, the app will be run immediately after the registration. Otherwise not.
        '''
        self.apps[app.status.name]=app
        app.dependences=self.preloads+app.dependences
        if run:
            self.log.open()
            name=app.status.name
            self.clock.add(name=name)
            with self.clock.get(name):
                cmp=app.status<=self.status
                if not (app.status.info and cmp):
                    if not cmp:self.update(**app.status.data)
                    if app.prepare is not None:app.prepare(self,app)
                    if app.run is not None:app.run(self,app)
                    app.status.info=True
                    app.status.update(self.status.data)
            self.log<<'App %s(%s): time consumed %ss.\n\n'%(name,app.__class__.__name__,self.clock.time(name))
            self.log.close()

    def rundependences(self,name):
        '''
        This method runs the dependences of the app specified by name.

        Parameters
        ----------
        name : any hashable object
            The name to specify whose dependences to be run.
        '''
        if name in self.apps:
            for app in self.apps[name].dependences:
                cmp=self.status<=app.status
                if not (app.status.info and cmp):
                    if not cmp:app.status.update(self.status.data)
                    if app.prepare is not None:app.prepare(self,app)
                    if app.run is not None:app.run(self,app)
                    app.status.info=True
        else:
            warn('%s rundependences warning: app(%s) not registered.'%(self.__class__.__name__,name))

    def summary(self):
        '''
        Generate the app report.
        '''
        self.log.open()
        self.log<<'Summary of %s(%s)'%(self.status.name,self.__class__.__name__)<<'\n'
        self.clock.record()
        self.log<<self.clock.tostr(form='s')<<'\n'
        self.log<<'\n'
        self.log.close()

class App(object):
    '''
    This class is the base class for those implementing specific tasks based on Engines.

    Attributes
    ----------
    status : Status
        The status of the app.
    dependences : list of App
        The apps on which this app depends.
    plot : logical
        A flag to tag whether the results are to be plotted.
    show : logical
        A flag to tag whether the plotted graph is to be shown.
    suspend : logical
        A flag to tag whether the program is suspended when the graph is plotted.
    parallel : logical
        A flag to tag whether the calculating process is to be paralleled.
    np : integer
        The number of processes used in parallel computing and 0 means the available maximum.
    save_data : logical
        A flag to tag whether the generated data of the result is to be saved on the hard drive.
    save_fig : logical
        A flag to tag whether the plotted graph to be saved.
    prepare : function
        The function called by the engine before it carries out the tasks.
    run : function
        The function called by the engine to carry out the tasks.
    '''
    SUSPEND_TIME=2

    def __new__(cls,*arg,**karg):
        '''
        This method automatically initialize the attributes of an App instance.
        '''
        result=object.__new__(cls)
        result.status=Status(name=karg.get('name',id(result)),info=False)
        result.status.update(karg.get('parameters',{}))
        attr_def={'dependences':[],'plot':True,'show':True,'suspend':False,'parallel':False,'np':0,'save_data':True,'save_fig':True,'prepare':None,'run':None}
        for key,value in attr_def.items():
            setattr(result,key,karg.get(key,value))
        return result
