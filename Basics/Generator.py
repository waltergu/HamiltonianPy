'''
======================
Generator of operators
======================

This module defines the core class of the `Basics` subpackage: `Generator`.
'''

__all__=['Generator']

from Constant import *
from Operator import *
from numpy import *
from numpy.linalg import norm
from collections import OrderedDict

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

    def __init__(self,bonds,config,table=None,terms=None,dtype=complex128,**options):
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
                if nv is not None and norm(array(nv)-array(term.value))>RZERO:
                    term.value=nv
                    self.parameters['alter'][key]=nv
                    selects[key]=True
            for key,select in selects.iteritems():
                if select:
                    self.cache['alter'][key]=Operators()
                    for bond in self.bonds:
                        self.cache['alter'][key]+=self.terms['alter'][key].operators(bond,self.config,table=self.table,dtype=self.dtype,**self.options)
