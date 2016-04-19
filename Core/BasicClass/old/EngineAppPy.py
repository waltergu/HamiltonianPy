'''
Engine and App.
'''
from NamePy import *
import time
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
        waiting_list: list
            The names of apps waiting to be run.
        apps: dict
            This dict contains all apps added to this engine.
            Note not all apps are in the waiting list.
    '''

    def __new__(cls,*arg,**karg):
        '''
        This method automatically initialize the attributes of an Engine instance.
        '''
        result=object.__new__(cls,*arg,**karg)
        result.din=karg['din'] if 'din' in karg else '.'
        result.dout=karg['dout'] if 'dout' in karg else '.'
        result.name=Name(prefix=karg['name'],suffix=result.__class__.__name__) if 'name' in karg else Name(suffix=result.__class__.__name__)
        if 'parameters' in karg:
            result.name.update(const=karg['parameters'])
        result.waiting_list=[]
        result.apps={}
        return result

    def __init__(self,*arg,**karg):
        pass

    def addapps(self,name=None,app=None):
        '''
        This method adds an app to self's attribute apps.
        Parameters:
            name: string,optional
                The name of the app.
                When this parameter is not None, it will be used as the key in self.apps. Whereas the app's class name will be used instead.
            app: App
                The app to be added.
                Only when the parameter name is not None will this apps goes into self.waiting_list.
        '''
        if name!=None:
            self.apps[name]=app
            self.waiting_list.append(name)
        else:
            self.apps[app.__class__.__name__]=app

    def runapps(self,name=None,clock=False):
        '''
        This method can be used in two different ways:
        1) self.runapps(name=...,clock=...)
            In this case, the app specified by the parameter name will be run.
        2) self.rundapps()
            In this case, the apps specified by those in self.waiting_list will be run.
        Parameters:
            name: string,optional
                The name to specify the app to be run.
            clock: logical, optional
                A flag to tell the program whether or not to record the time each run app consumed.
                Note for case 2, this parameter is omitted since the time each app in self.waiting_list costs is automatically recorded.
        '''
        if name!=None:
            if clock :
                stime=time.time()
                self.apps[name].run(self,self.apps[name])
                etime=time.time()
                print 'App '+name+': time consumed '+str(etime-stime)+'s.'
            else:
                self.apps[name].run(self,self.apps[name])
        else:
            for name in self.waiting_list:
                stime=time.time()
                self.apps[name].run(self,self.apps[name])
                etime=time.time()
                print 'App '+name+': time consumed '+str(etime-stime)+'s.'

class App(object):
    '''
    This class is the base class for those implementing specific tasks based on Engines.
    Attributes:
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
    '''
    
    def __new__(cls,*arg,**karg):
        '''
        This method automatically initialize the attributes of an App instance.
        '''
        result=object.__new__(cls,*arg,**karg)
        result.plot=karg['plot'] if 'plot' in karg else True
        result.show=karg['show'] if 'show' in karg else True
        result.parallel=karg['parallel'] if 'parallel' in karg else False
        result.np=karg['np'] if 'np' in karg else 0
        result.save_data=karg['save_data'] if 'save_data' in karg else True
        if 'run' in karg: result.run=karg['run']
        return result

    def __init__(self,*arg,**karg):
        pass
