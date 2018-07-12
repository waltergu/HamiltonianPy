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
    boundaries : 4-tuple in the form (names,values,vectors,function)
        * names : tuple of str
            The variable names of the boundary phases.
        * values : tuple of number
            The variable values of the boundary phases.
        * vectors : 2d ndarray like
            The translation vectors of the lattice.
        * function : callable
            The function used to transform the boundary operators.
    _operators_ : dict
        The cache to handle the generation and update of the operators.
    _matrix_ : dict
        The cache to handle the generation and update of the matrix representation of the operators.
    dtype : float64 or complex128
        The data type of the coefficients of the operators.
    options : dict
        The extra key word arguments for Term.operators.
    '''

    def __init__(self,bonds,config,table=None,terms=None,boundaries=None,dtype=np.complex128,**options):
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
        boundaries : 4-tuple in the form (names,values,vectors,function)
            * names : tuple of str
                The variable names of the boundary phases.
            * values : tuple of number
                The variable values of the boundary phases.
            * vectors : 2d ndarray like
                The translation vectors of the lattice.
            * function : callable
                The function used to transform the boundary operators.
        dtype : numpy.float64, numpy.complex128, optional
            The data type of the coefficients of the generated operators.
        options : dict, optional
            The extra key word arguments for Term.operators.
        '''
        self.bonds=bonds
        self.config=config
        self.table=table
        self.boundaries=boundaries
        self.dtype=dtype
        self.options=options
        self.set_terms(terms)
        self.set_operators()
        self._matrix_={}

    def reset(self,bonds=None,config=None,table=None,terms=None,boundaries=None,dtype=None,**options):
        '''
        Refresh the generator.

        Parameters
        ----------
        bonds,config,table,terms,boundaries,dtype,options : see Generator.__init__ for details.
        '''
        if bonds is not None: self.bonds=bonds
        if config is not None: self.config=config
        if table is not None: self.table=table
        if dtype is not None: self.dtype=dtype
        if boundaries is not None: self.boundaries=boundaries
        if len(options)>0: self.options=options
        if terms is not None: self.set_terms(terms)
        self.set_operators()
        self._matrix_={}

    def set_terms(self,terms):
        '''
        Set the terms of the generator.
        '''
        self.terms={'const':[],'alter':[]}
        if terms is not None:
            for term in terms:
                if term.modulate is None:
                    self.terms['const'].append(term)
                else:
                    self.terms['alter'].append(term)

    def set_operators(self):
        '''
        Set the cache of the operators.
        '''
        self._operators_={'const':Operators(),'alter':[],'bound':Operators()}
        for term in self.terms['const']:
            for bond in self.bonds:
                opts=term.operators(bond,self.config,table=self.table,dtype=self.dtype,**self.options)
                if self.boundaries is None or bond.isintracell():
                    self._operators_['const']+=opts
                else:
                    values,vectors,function=self.boundaries[1:]
                    for opt in opts.itervalues():
                        self._operators_['bound']+=function(opt,vectors,values)
        for term in self.terms['alter']:
            operators=Operators()
            for bond in self.bonds:
                opts=term.operators(bond,self.config,table=self.table,dtype=self.dtype,**self.options)
                if self.boundaries is None or bond.isintracell():
                    operators+=opts
                else:
                    values,vectors,function=self.boundaries[1:]
                    for opt in opts.itervalues():
                        self._operators_['bound']+=function(opt,vectors,values)
            self._operators_['alter'].append(operators)

    def set_matrix(self,sector,optrep,*args,**kargs):
        '''
        Set the cache of the matrix representation of the operators.

        Parameters
        ----------
        sector : str
            The sector of the matrix representation of the operators.
        optrep : callable
            The function to generate the matrix representation of a single operator.
        args,kargs : optional
            The extra arguments of the function `optrep`.
        '''
        self._matrix_[sector]={'const':0,'alter':[],'bound':[optrep,args,kargs]}
        for operator in self._operators_['const'].itervalues():
            self._matrix_[sector]['const']+=optrep(operator,*args,**kargs)
        for term in self.terms['alter']:
            term,matrix=term.unit,0
            for bond in self.bonds:
                for operator in term.operators(bond,self.config,table=self.table,dtype=self.dtype,**self.options).itervalues():
                    matrix+=optrep(operator,*args,**kargs)
            self._matrix_[sector]['alter'].append(matrix)

    def __str__(self):
        '''
        Convert an instance to string.
        '''
        result=[]
        result.append('Generator(')
        result.append('terms=(%s'%(', '.join('%s'%term for term in self.terms['const'])))
        result.append('%s%s='%(', '.join('%s'%term for key,term in self.terms['alter']),')'))
        result.append(')')
        return ', '.join(result)

    @property
    def operators(self):
        '''
        This method returns all the generated operators, both constant ones and alterable ones.
        '''
        result=Operators()
        result.update(self._operators_['const'])
        for opts in self._operators_['alter']:
            result+=opts
        result+=self._operators_['bound']
        return result

    def matrix(self,sector):
        '''
        This method returns the matrix representation of the operators.

        Parameters
        ----------
        sector : str
            The sector of the matrix representation of the operators.

        Returns
        -------
        matrix-like
            The matrix representation of the operators.
        '''
        result=0
        result+=self._matrix_[sector]['const']
        for term,matrix in zip(self.terms['alter'],self._matrix_[sector]['alter']):
            result+=matrix*term.value
        for opt in self._operators_['bound'].itervalues():
            optrep,args,kargs=self._matrix_[sector]['bound']
            result+=optrep(opt,*args,**kargs)
        return result

    def update(self,**karg):
        '''
        This method updates the alterable operators by keyword arguments.
        '''
        if self.boundaries is not None:
            self._operators_['bound']=Operators()
            names,values,vectors,function=self.boundaries
            thetas=[karg.get(name,value) for name,value in zip(names,values)]
            for term in self.terms['const']:
                for bond in self.bonds:
                    if not bond.isintracell():
                        opts=term.operators(bond,self.config,table=self.table,dtype=self.dtype,**self.options)
                        for opt in opts.itervalues(): self._operators_['bound']+=function(opt,vectors,thetas)
            self.boundaries=(names,thetas,vectors,function)
        for pos,term in enumerate(self.terms['alter']):
            nv=term.modulate(**karg)
            if nv is not None and nl.norm(np.array(nv)-np.array(term.value))>RZERO:
                term.value=nv
                self._operators_['alter'][pos]=Operators()
                for bond in self.bonds:
                    opts=term.operators(bond,self.config,table=self.table,dtype=self.dtype,**self.options)
                    if self.boundaries is None or bond.isintracell():
                        self._operators_['alter'][pos]+=opts
                    else:
                        for opt in opts.itervalues(): self._operators_['bound']+=function(opt,vectors,thetas)

    def view(self,bondselect=None,termselect=None,pidon=True,bonddr='+',show=True,suspend=False,close=True):
        '''
        View the index packs of the terms on the bonds.

        Parameters
        ----------
        bondselect : callable, optional
            The select function of the bonds.
        termselect : callable, optional
            The select function of the terms.
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
            if bondselect is None or bondselect(bond):
                assert bonddr in ('+','-')
                (x,y),(dx,dy)=(bond.spoint.rcoord,bond.rcoord) if bonddr=='+' else (bond.epoint.rcoord,bond.reversed.rcoord)
                if nl.norm(bond.rcoord)>RZERO: plt.arrow(x,y,dx,dy,ls='--' if nl.norm(bond.icoord)>RZERO else '-',lw=2,color='red',length_includes_head=True,alpha=0.5)
                packs=[term.strrep(bond if bonddr=='+' else bond.reversed,self.config) for term in it.chain(self.terms['const'],self.terms['alter']) if termselect is None or termselect(term)]
                if len(packs)>0:
                    plt.text(x+dx/2,y+dy/2,'\n'.join(sorted(packs,key=len)),color='green',horizontalalignment='center',verticalalignment='center',fontproperties=font)
        plt.xlim([xmin-(xmax-xmin)*0.30,xmax+(xmax-xmin)*0.30])
        plt.ylim([ymin-(ymax-ymin)*0.30,ymax+(ymax-ymin)*0.30])
        if show and suspend: plt.show()
        if show and not suspend: plt.pause(1)
        if close: plt.close()
