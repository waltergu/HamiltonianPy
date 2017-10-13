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
    preloads : list of App
        The preloaded apps of the engine, which are the common dependences of all the other apps registered on it.
    apps : dict of App
        The apps registered on this engine (the dependences of the apps not included).
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
        result.preloads=karg.get('preloads',[])
        result.apps={}
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
        self.apps[app.name]=app
        app.dependences=self.preloads+app.dependences
        if run:
            self.log.open()
            self.clock.add(name=app.name)
            with self.clock.get(app.name):
                match=app.parameters.match(self.parameters)
                if app.virgin or not match:
                    if not match: self.update(**app.parameters)
                    if app.prepare is not None: app.prepare(self,app)
                    if app.run is not None: app.run(self,app)
                    app.virgin=False
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
        if name in self.apps:
            for app in self.apps[name].dependences:
                match=self.parameters.match(app.parameters)
                if app.virgin or not match:
                    app.parameters.update(self.parameters)
                    if app.prepare is not None: app.prepare(self,app)
                    if app.run is not None: app.run(self,app)
                    app.virgin=False
        else:
            warn('%s rundependences warning: app(%s) not registered.'%(self.__class__.__name__,name))

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
        True when the app has not been run, otherwise False.
    np : integer
        The number of processes used in parallel computing and 0 means the available maximum.
    plot : logical
        A flag to tag whether the results are to be plotted.
    show : logical
        A flag to tag whether the plotted graph is to be shown.
    suspend : logical
        A flag to tag whether the program is suspended when the graph is plotted.
    savefig : logical
        A flag to tag whether the plotted graph to be saved.
    savedata : logical
        A flag to tag whether the generated data of the result is to be saved on the hard drive.
    dependences : list of App
        The apps on which this app depends.
    update : callable
        The function called by the engine to update the app's attributes before it prepares the tasks.
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
        result.dependences=karg.get('dependences',[])
        result.update=karg.get('update',None)
        result.prepare=karg.get('prepare',None)
        result.run=karg.get('run',None)
        return result

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
                if 'legend' in options: plt.legend(shadow=True,fancybox=True,loc=options.get('legendloc',None))
            else:
                plt.plot(data[:,0],data[:,1:])
                if 'legend' in options: plt.legend(options['legend'],shadow=True,fancybox=True,loc=options.get('legendloc',None))
        elif mode=='P':
            plt.colorbar(plt.pcolormesh(data[:,:,0],data[:,:,1],data[:,:,2]))
            if 'axis' in options: plt.axis(options.get('axis','equal'))
        if self.show and self.suspend: plt.show()
        if self.show and not self.suspend: plt.pause(App.SUSPEND_TIME)
        if self.savefig: plt.savefig('%s.png'%name)
        plt.close()
