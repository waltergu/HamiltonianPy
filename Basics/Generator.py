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
    This class provides methods to generate and update operators, and optionally their matrix representations, according to terms, bonds and configuration of internal degrees of freedom.

    Attributes
    ---------
    bonds : list of Bond
        The bonds of the model.
    config : IDFConfig
        The configuration of the internal degrees of freedom.
    table : Table
        The index-sequence table of the system
    terms : dict
        It contains all terms contained in the model, divided into two groups, the constant ones and the alterable ones.
    _operators_ : dict
        The cache to handle the generation and update of the operators.
    _matrix_ : dict
        The cache to handle the generation and update of the matrix representation of the operators.
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
        self.set_terms_operators_(terms)

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
        if len(options)>0: self.options=options
        if terms is not None: self.set_terms_operators_(terms)

    def set_terms_operators_(self,terms):
        '''
        Set the terms and _operators_ of the generator.
        '''
        self.terms={'const':OrderedDict(),'alter':OrderedDict()}
        self._operators_={'const':Operators(),'alter':OrderedDict()}
        if terms is not None:
            for term in terms:
                if term.modulate is not None:
                    self.terms['alter'][term.id]=term
                    self._operators_['alter'][term.id]=Operators()
                    for bond in self.bonds:
                        self._operators_['alter'][term.id]+=term.operators(bond,self.config,table=self.table,dtype=self.dtype,**self.options)
                else:
                    self.terms['const'][term.id]=term
                    for bond in self.bonds:
                        self._operators_['const']+=term.operators(bond,self.config,table=self.table,dtype=self.dtype,**self.options)

    def __str__(self):
        '''
        Convert an instance to string.
        '''
        result=[]
        result.append('Generator(')
        result.append('terms=(%s'%(', '.join(['%s:%s'%(key,term) for key,term in self.terms['const'].iteritems()])))
        result.append('%s%s='%(', '.join(['%s:%s'%(key,term) for key,term in self.terms['alter'].iteritems()]),')'))
        result.append(')')
        return ', '.join(result)

    @property
    def parameters(self):
        '''
        The parameters of the generator.
        '''
        result={'const':OrderedDict(),'alter':OrderedDict()}
        for key,term in self.terms['const'].iteritems():
            result['const'][key]=term.value
        for key,term in self.terms['alter'].iteritems():
            result['alter'][key]=term.value
        return result

    @property
    def operators(self):
        '''
        This method returns all the generated operators, both constant ones and alterable ones.
        '''
        result=Operators()
        result.update(self._operators_['const'])
        for opts in self._operators_['alter'].itervalues():
            result+=opts
        return result

    @property
    def matrix(self):
        '''
        This method returns the matrix representation of the operators.
        '''
        result=0
        result+=self._matrix_['const']
        for term,matrix in zip(self.terms['alter'].itervalues(),self._matrix_['alter'].itervalues()):
            result+=matrix*term.value
        return result

    def update(self,**karg):
        '''
        This method updates the alterable operators by keyword arguments.
        '''
        selects={key:False for key in self.terms['alter'].iterkeys()}
        for key,term in self.terms['alter'].iteritems():
            nv=term.modulate(**karg)
            if nv is not None and nl.norm(np.array(nv)-np.array(term.value))>RZERO:
                term.value=nv
                selects[key]=True
        for key,select in selects.iteritems():
            if select:
                self._operators_['alter'][key]=Operators()
                for bond in self.bonds:
                    self._operators_['alter'][key]+=self.terms['alter'][key].operators(bond,self.config,table=self.table,dtype=self.dtype,**self.options)

    def refresh(self,optrep,*args,**kargs):
        '''
        Refresh the cache of the matrix representation of the operators.

        Parameters
        ----------
        optrep : callable
            The function to generate the matrix representation of a single operator.
        args,kargs : optional
            The extra arguments of the function `optrep`.
        '''
        self._matrix_={'const':0,'alter':OrderedDict()}
        for operator in self._operators_['const'].itervalues():
            self._matrix_['const']+=optrep(operator,*args,**kargs)
        for key,term in self.terms['alter'].iteritems():
            term=term.unit
            self._matrix_['alter'][key]=0
            for bond in self.bonds:
                for operator in term.operators(bond,self.config,table=self.table,dtype=self.dtype,**self.options).itervalues():
                    self._matrix_['alter'][key]+=optrep(operator,*args,**kargs)

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
