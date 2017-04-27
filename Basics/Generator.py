'''
Generator.
'''

__all__=['Generator']

from Constant import *
from Operator import *
from numpy import *
from numpy.linalg import norm
from collections import OrderedDict

class Generator(object):
    '''
    This class provides methods to generate and update operators according to terms, bonds and configuration of degrees of freedom.

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
        The working space used to handle the generation and update of the operators.
    dtype : float64 or complex128
        The data type of the coefficients of the operators.
    '''

    def __init__(self,bonds,config,table=None,terms=None,dtype=complex128):
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
            The terms whose corresponding operators are to be generated and updated.
            Those terms having the attribute modulate will go into self.terms['alter'] and the others will go into self.terms['const'].
        dtype : numpy.float64, numpy.complex128
            The data type of the coefficients of the generated operators.
        '''
        self.bonds=bonds
        self.config=config
        self.table=table
        self.dtype=dtype
        self.set_parameters_and_terms(terms)
        self.set_cache()

    def reset(self,bonds=None,config=None,table=None,terms=None,dtype=None):
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
            The new terms whose corresponding operators are to be generated and updated.
        dtype : numpy.float64, numpy.complex128
            The new data type of the coefficients of the generated operators.
        '''
        if bonds is not None: self.bonds=bonds
        if config is not None: self.config=config
        if table is not None: self.table=table
        if dtype is not None: self.dtype=dtype
        if terms is not None: self.set_parameters_and_terms(terms)
        self.set_cache()

    def set_parameters_and_terms(self,terms):
        self.parameters={}
        self.terms={}
        self.parameters['const']=OrderedDict()
        self.parameters['alter']=OrderedDict()
        self.terms['const']={}
        self.terms['alter']={}
        if terms is not None:
            for term in terms:
                if hasattr(term,'modulate'):
                    self.parameters['alter'][term.id]=term.value
                    self.terms['alter'][term.id]=+term
                else:
                    self.parameters['const'][term.id]=term.value
                    lterm=+term
                    if lterm.__class__.__name__ in self.terms['const']:
                        self.terms['const'][lterm.__class__.__name__].extend(lterm)
                    else:
                        self.terms['const'][lterm.__class__.__name__]=lterm

    def set_cache(self):
        self.cache={}
        if 'const' in self.terms:
            self.cache['const']=OperatorCollection()
            for bond in self.bonds:
                for terms in self.terms['const'].itervalues():
                    self.cache['const']+=terms.operators(bond,self.config,table=self.table,dtype=self.dtype)
        if 'alter' in self.terms:
            self.cache['alter']={}
            for key in self.terms['alter'].iterkeys():
                self.cache['alter'][key]=OperatorCollection()
                for bond in self.bonds:
                    self.cache['alter'][key]+=self.terms['alter'][key].operators(bond,self.config,table=self.table,dtype=self.dtype)

    def __str__(self):
        '''
        Convert an instance to string.
        '''
        result=[]
        result.append('Generator(parameters=%s'.self.parameters)
        result.append('terms=(%s'%(', '.join(['%s:%s'%(key,obj) for key,obj in self.terms['const'].iteritems()])))
        result.append('%s%s='%(', '.join(['%s:%s'%(key,obj) for key,obj in self.terms['alter'].iteritems()]),')'))
        result.append(')')
        return ', '.join(result)

    @property
    def operators(self):
        '''
        This method returns all the operators generated by self.
        '''
        result=OperatorCollection()
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
                nv=term[0].modulate(**karg)
                if nv is not None and norm(array(nv)-array(term[0].value))>RZERO:
                    term[0].value=nv
                    self.parameters['alter'][key]=nv
                    selects[key]=True
            for key,select in selects.iteritems():
                if select:
                    self.cache['alter'][key]=OperatorCollection()
                    for bond in self.bonds:
                        self.cache['alter'][key]+=self.terms['alter'][key].operators(bond,self.config,table=self.table,dtype=self.dtype)
