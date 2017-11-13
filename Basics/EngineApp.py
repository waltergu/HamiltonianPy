'''
==============
Engine and app
==============

This module defines the general framework to apply algorithms to tasks, including:
    * classes: Parameters, Engine, App
'''

__all__=['Parameters','Engine','App']

import os
import numpy as np
import itertools as it
import matplotlib.pyplot as plt
from numpy.linalg import norm
from scipy import interpolate
from collections import OrderedDict
from Utilities import RZERO,Log,Timers,decimaltostr
from warnings import warn

class Parameters(OrderedDict):
    '''
    The parameters of an Engine/App.
    '''

    def match(self,other):
        '''
        Judge whether a set of parameters matches another. True when their shared entries have the same values, otherwise False.
        '''
        for key,value in self.iteritems():
            if key in other:
                if norm(np.array(value)-np.array(other[key]))>RZERO:
                    return False
        else:
            return True

    def __str__(self):
        '''
        Convert an instance to string.
        '''
        return '_'.join(decimaltostr(value,Engine.NDECIMAL) for key,value in self.iteritems())

class Engine(object):
    '''
    This class is the base class for all Hamiltonian-oriented classes.

    Attributes
    ----------
    din : str
        The directory where the engine reads data.
    dout : str
        The directory where the engine writes data.
    log : Log
        The log of the engine.
    name : str
        The name of the engine.
    parameters : Parameters
        The parameters of the engine.
    map : callable
        This function maps a set of parameters to another.
    preloads : list of str
        The names of the preloaded apps of the engine.
    apps : dict of App
        The apps of this engine.
    records : dict
        The records of the returned data of the apps of this engine.
    clock : Timers
        The clock of the engine.
    '''
    DEBUG=True
    MKDIR=True
    NDECIMAL=5

    def __new__(cls,*arg,**karg):
        '''
        This method automatically initialize the attributes of an Engine instance.
        '''
        result=object.__new__(cls)
        result.din=karg.get('din','.')
        result.dout=karg.get('dout','.')
        result.log=Log(dir=karg.get('dlog','.'))
        for dir in [result.din,result.dout,result.log.dir]:
            if cls.MKDIR and not os.path.exists(dir): os.makedirs(dir)
        result.name=karg.get('name','')
        result.parameters=Parameters(karg.get('parameters',()))
        result.map=karg.get('map',None)
        result.clock=Timers()
        result.preloads=[]
        result.apps={}
        result.records={}
        return result

    def data(self,parameters):
        '''
        The data of the engine.
        '''
        return self.map(parameters) if callable(self.map) else parameters

    def logging(self):
        '''
        Set the log of the engine.
        '''
        if not self.DEBUG: self.log.reset('%s.log'%self)

    def tostr(self,mask=()):
        '''
        Get the engine's string representation.

        Parameters
        ----------
        mask : tuple of str
            The mask of the object's data.

        Returns
        -------
        str
            The engine's string representation.
        '''
        result=[]
        result.append(self.name)
        result.append('_'.join(decimaltostr(value,Engine.NDECIMAL) for key,value in self.parameters.iteritems() if key not in mask))
        result.append(self.__class__.__name__)
        return '_'.join(result)

    def __str__(self):
        '''
        Convert an instance to string.
        '''
        return self.tostr()

    def update(self,**paras):
        '''
        This method update the engine.
        '''
        for key,value in paras.iteritems():
            if key in self.parameters: self.parameters[key]=value

    def add(self,app):
        '''
        This method add a new app to the engine.

        Parameters
        ----------
        app : App
            The app to be added to the engine.
        '''
        self.apps[app.name]=app
        self.records[app.name]=None

    def preload(self,app):
        '''
        This method preload an app onto the engine.

        Parameters
        ----------
        app : App
            The app to be preloaded.
        '''
        self.add(app)
        self.preloads.append(app.name)

    def register(self,app):
        '''
        This method register a new app on the engine.

        Parameters
        ----------
        app : App
            The app to be registered on this engine.
        '''
        self.add(app)
        self.log.open()
        self.clock.add(name=app.name)
        with self.clock.get(app.name):
            match=app.parameters.match(self.parameters)
            if app.virgin or not match:
                if not match: self.update(**app.parameters)
                if app.prepare is not None: app.prepare(self,app)
                if app.run is not None: self.records[app.name]=app.run(self,app)
                app.virgin=False
                app.update(**self.parameters)
                app.parameters.update(self.parameters)
        self.log<<'App %s(%s): time consumed %ss.\n\n'%(app.name,app.__class__.__name__,self.clock.time(app.name))
        self.log.close()

    def rundependences(self,name):
        '''
        This method runs the dependences of the app specified by name.

        Parameters
        ----------
        name : any hashable object
            The name to specify whose dependences to be run.
        '''
        for app in it.chain(self.preloads,self.apps[name].dependences):
            app=self.apps[app]
            match=self.parameters.match(app.parameters)
            if app.virgin or not match:
                app.update(**self.parameters)
                app.parameters.update(self.parameters)
                if app.prepare is not None: app.prepare(self,app)
                if app.run is not None: self.records[app.name]=app.run(self,app)
                app.virgin=False

    def summary(self):
        '''
        Generate the app report.
        '''
        self.log.open()
        self.log<<'Summary of %s(%s)\n'%(self.name,self.__class__.__name__)
        self.clock.record()
        self.log<<self.clock.tostr(form='s')<<'\n'
        self.log<<'\n'
        self.log.close()

class App(object):
    '''
    This class is the base class for those implementing specific tasks based on Engines.

    Attributes
    ----------
    name : str
        The name of the app.
    parameters : Parameters
        The parameters of the app.
    virgin : logical
        True when the app has not been run. Otherwise False.
    np : integer
        The number of processes will be used in parallel computing. 0 means the available maximum.
    plot : logical
        When True, the results will be be plotted. Otherwise not.
    show : logical
        When True, the plotted graph will be shown. Otherwise not.
    suspend : logical
        When True, the program is suspended when the graph is plotted. Otherwise not.
    savefig : logical
        When True, the plotted graph will be saved. Otherwise not.
    savedata : logical
        When True, the results will be saved on the hard drive. Otherwise not.
    returndata: logical
        When True, the results will be returned. Otherwise not.
    dependences : list of str
        The names of the apps that this app depends on.
    map : callable
        The function that maps the a set of parameters to the app's attributes.
    prepare : callable
        The function called by the engine before it carries out the tasks.
    run : callable
        The function called by the engine to carry out the tasks.
    '''
    SUSPEND_TIME=2

    def __new__(cls,*arg,**karg):
        '''
        This method automatically initialize the attributes of an App instance.
        '''
        result=object.__new__(cls)
        result.name=karg.get('name',str(id(result)))
        result.parameters=Parameters(karg.get('parameters',()))
        result.virgin=True
        result.np=karg.get('np',None)
        result.plot=karg.get('plot',True)
        result.show=karg.get('show',True)
        result.suspend=karg.get('suspend',False)
        result.savefig=karg.get('savefig',True)
        result.savedata=karg.get('savedata',True)
        result.returndata=karg.get('returndata',True)
        result.dependences=karg.get('dependences',[])
        result.map=karg.get('map',None)
        result.prepare=karg.get('prepare',None)
        result.run=karg.get('run',None)
        return result

    def update(self,**karg):
        '''
        Update the attributes of the app.
        '''
        if callable(self.map) and len(karg)>0:
            for key,value in self.map(karg).iteritems():
                assert hasattr(self,key)
                setattr(self,key,value)

    def figure(self,mode,data,name,**options):
        '''
        Generate a figure to view the data.

        Parameters
        ----------
        mode : 'L','P'
            'L' for lines and 'P' for pcolor.
        data : ndarray
            The data to be viewed.
        name : str
            The name of the figure.
        options : dict
            Other options.
        '''
        assert mode in ('L','P')
        plt.title(os.path.basename(name))
        if mode=='L':
            if options.get('interpolate',False):
                plt.plot(data[:,0],data[:,1],'r.')
                X=np.linspace(data[:,0].min(),data[:,0].max(),10*data.shape[0])
                for i in xrange(1,data.shape[1]):
                    tck=interpolate.splrep(data[:,0],data[:,i],k=3)
                    Y=interpolate.splev(X,tck,der=0)
                    plt.plot(X,Y,label=options['legend'][i-1] if 'legend' in options else None)
                if 'legend' in options:
                    leg=plt.legend(fancybox=True,loc=options.get('legendloc',None))
                    leg.get_frame().set_alpha(0.5)
            else:
                plt.plot(data[:,0],data[:,1:])
                if 'legend' in options:
                    leg=plt.legend(options['legend'],fancybox=True,loc=options.get('legendloc',None))
                    leg.get_frame().set_alpha(0.5)
        elif mode=='P':
            plt.colorbar(plt.pcolormesh(data[:,:,0],data[:,:,1],data[:,:,2]))
            if 'axis' in options: plt.axis(options.get('axis','equal'))
        if self.show and self.suspend: plt.show()
        if self.show and not self.suspend: plt.pause(App.SUSPEND_TIME)
        if self.savefig: plt.savefig('%s.png'%name)
        plt.close()
