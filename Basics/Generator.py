'''
======================
Generator of operators
======================

This module defines the core class of the `Basics` subpackage: `Generator`.
'''

__all__=['Generator']

from Utilities import RZERO
from Operator import *
from collections import OrderedDict
from matplotlib.font_manager import FontProperties
import numpy as np
import itertools as it
import numpy.linalg as nl
import matplotlib.pyplot as plt

class Generator(object):
    '''
    This class provides methods to generate and update operators according to terms, bonds and configuration of internal degrees of freedom.

    Attributes
    ---------
    bonds : list of Bond
        The bonds of the model.
    config : IDFConfig
        The configuration of the internal degrees of freedom.
    table : Table
        The index-sequence table of the system
    parameters : dict
        It contains all model parameters, divided into two groups, the constant ones and the alterable ones.
    terms : dict
        It contains all terms contained in the model, divided into two groups, the constant ones and the alterable ones.
    cache : dict
        The cache to handle the generation and update of the operators.
    dtype : float64 or complex128
        The data type of the coefficients of the operators.
    options : dict
        The extra key word arguments for Term.operators.
    '''

    def __init__(self,bonds,config,table=None,terms=None,dtype=np.complex128,**options):
        '''
        Constructor.

        Parameters
        ----------
        bonds : list of Bond
            The bonds.
        config : IDFConfig
            The configuration of the internal degrees of freedom.
        table : Table, optional
            The index-sequence table.
        terms : list of Term, optional
            The terms whose operators are to be generated and updated.
        dtype : numpy.float64, numpy.complex128, optional
            The data type of the coefficients of the generated operators.
        options : dict, optional
            The extra key word arguments for Term.operators.
        '''
        self.bonds=bonds
        self.config=config
        self.table=table
        self.dtype=dtype
        self.options=options
        self.set_parameters_and_terms(terms)
        self.set_cache()

    def reset(self,bonds=None,config=None,table=None,terms=None,dtype=None,**options):
        '''
        Refresh the generator.

        Parameters
        ----------
        bonds : list of Bond, optional
            The new bonds.
        config : IDFConfig, optional
            The new configuration of the internal degrees of freedom.
        table : Table, optional
            The new index-sequence table.
        terms : list of Term, optional
            The new terms whose operators are to be generated and updated.
        dtype : numpy.float64, numpy.complex128, optional
            The new data type of the coefficients of the generated operators.
        options : dict, optional
            The extra key word arguments for Term.operators.
        '''
        if bonds is not None: self.bonds=bonds
        if config is not None: self.config=config
        if table is not None: self.table=table
        if dtype is not None: self.dtype=dtype
        if terms is not None: self.set_parameters_and_terms(terms)
        if len(options)>0: self.options=options
        self.set_cache()

    def set_parameters_and_terms(self,terms):
        '''
        Set the parameters and terms of the generator.
        '''
        self.parameters={'const':OrderedDict(),'alter':OrderedDict()}
        self.terms={'const':{},'alter':{}}
        if terms is not None:
            for term in terms:
                if term.modulate is not None:
                    self.parameters['alter'][term.id]=term.value
                    self.terms['alter'][term.id]=term
                else:
                    self.parameters['const'][term.id]=term.value
                    self.terms['const'][term.id]=term

    def set_cache(self):
        self.cache={}
        if 'const' in self.terms:
            self.cache['const']=Operators()
            for bond in self.bonds:
                for term in self.terms['const'].itervalues():
                    self.cache['const']+=term.operators(bond,self.config,table=self.table,dtype=self.dtype,**self.options)
        if 'alter' in self.terms:
            self.cache['alter']={}
            for key,term in self.terms['alter'].iteritems():
                self.cache['alter'][key]=Operators()
                for bond in self.bonds:
                    self.cache['alter'][key]+=term.operators(bond,self.config,table=self.table,dtype=self.dtype,**self.options)

    def __str__(self):
        '''
        Convert an instance to string.
        '''
        result=[]
        result.append('Generator(parameters=%s'%self.parameters)
        result.append('terms=(%s'%(', '.join(['%s:%s'%(key,term) for key,term in self.terms['const'].iteritems()])))
        result.append('%s%s='%(', '.join(['%s:%s'%(key,term) for key,term in self.terms['alter'].iteritems()]),')'))
        result.append(')')
        return ', '.join(result)

    @property
    def operators(self):
        '''
        This method returns all the generated operators, both constant ones and alterable ones.
        '''
        result=Operators()
        if 'const' in self.cache:
            result.update(self.cache['const'])
        if 'alter' in self.cache:
            for opts in self.cache['alter'].itervalues():
                result+=opts
        return result

    def update(self,**karg):
        '''
        This method updates the alterable operators by keyword arguments.
        '''
        if 'alter' in self.terms:
            selects={key:False for key in self.terms['alter'].iterkeys()}
            for key,term in self.terms['alter'].iteritems():
                nv=term.modulate(**karg)
                if nv is not None and nl.norm(np.array(nv)-np.array(term.value))>RZERO:
                    term.value=nv
                    self.parameters['alter'][key]=nv
                    selects[key]=True
            for key,select in selects.iteritems():
                if select:
                    self.cache['alter'][key]=Operators()
                    for bond in self.bonds:
                        self.cache['alter'][key]+=self.terms['alter'][key].operators(bond,self.config,table=self.table,dtype=self.dtype,**self.options)

    def view(self,bondmask=None,termmask=None,pidon=True,bonddr='+',show=True,suspend=False,close=True):
        '''
        View the index packs of the terms on the bonds.

        Parameters
        ----------
        bondmask : callable, optional
            The mask function of the bonds.
        termmask : callable, optional
            The mask function of the terms.
        pidon : logical, optional
            True for showing the pids of the points of the bonds.
        bonddr : '+'/'-', optional
            The direction of the bonds.
        show : logical, optional
            True for showing the view and False for not.
        suspend : logical, optional
            True for suspending the view and False for not.
        close : logical, optional
            True for closing the view and False for not.
        '''
        plt.axis('off')
        plt.axis('equal')
        xmax,xmin,ymax,ymin=0,0,0,0
        points,font=set(),FontProperties(style='italic',weight='bold',size='large')
        for bond in self.bonds:
            assert len(bond.rcoord)==2
            for i,point in enumerate([bond.spoint,bond.epoint]):
                pid=point.pid
                xmax,xmin=max(xmax,point.rcoord[0]),min(xmin,point.rcoord[0])
                ymax,ymin=max(ymax,point.rcoord[1]),min(ymin,point.rcoord[1])
                if pid not in points:
                    x,y=point.rcoord if i==0 else point.rcoord-bond.icoord
                    plt.scatter(x,y,zorder=2,alpha=0.5)
                    if pidon: plt.text(x,y,'%s%s'%('' if pid.scope is None else '%s*'%pid.scope,pid.site),color='blue',horizontalalignment='center',fontproperties=font)
                    points.add(point.pid)
            if bondmask is None or bondmask(bond):
                assert bonddr in ('+','-')
                (x,y),(dx,dy)=(bond.spoint.rcoord,bond.rcoord) if bonddr=='+' else (bond.epoint.rcoord,bond.reversed.rcoord)
                if nl.norm(bond.rcoord)>RZERO: plt.arrow(x,y,dx,dy,ls='--' if nl.norm(bond.icoord)>RZERO else '-',lw=2,color='red',length_includes_head=True,alpha=0.5)
                packs=[term.strrep(bond,self.config) for term in it.chain(self.terms['const'].values(),self.terms['alter'].values()) if termmask is None or termmask(term)]
                if len(packs)>0:
                    plt.text(x+dx/2,y+dy/2,'\n'.join(sorted(packs,key=len)),color='green',horizontalalignment='center',verticalalignment='center',fontproperties=font)
        plt.xlim([xmin-(xmax-xmin)*0.30,xmax+(xmax-xmin)*0.30])
        plt.ylim([ymin-(ymax-ymin)*0.30,ymax+(ymax-ymin)*0.30])
        if show and suspend: plt.show()
        if show and not suspend: plt.pause(1)
        if close: plt.close()
