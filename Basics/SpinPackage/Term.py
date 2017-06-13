'''
----------
Spin terms
----------

Spin terms, including:
    * classes: SpinTerm, SpinTerms
'''

__all__=['SpinTerm','SpinTerms']

from numpy import *
from ..Constant import *
from ..Term import *
from ..DegreeOfFreedom import *
from ..Operator import * 
from DegreeOfFreedom import *
from Operator import * 

class SpinTerm(Term):
    '''
    This class provides a complete and unified description for spin terms.

    Attributes
    ----------
    neighbour : integer
        The order of neighbour of this spin term.
    indexpacks : IndexPacks or function which returns IndexPacks
        The indexpacks of the spin term.
        When it is a function, it can return bond dependent indexpacks as needed.
    amplitude : function which returns float or complex
        This function can return bond dependent coefficient as needed.
    '''

    def __init__(self,id,value,neighbour,indexpacks,amplitude=None,modulate=None):
        '''
        Constructor.

        Parameters
        ----------
        id : string
            The specific id of the term.
        value : float or complex
            The overall coefficient of the term.
        neighbour : integer
            The order of neighbour of the term.
        indexpacks : IndexPacks or function
            When it is a function, it must return an instance of IndexPacks and take an instance of Bond as its only argument.
        amplitude : function, optional
            It must return a float or complex and take an instance of Bond as its only argument.
        modulate : function, optional
            It must return a float or complex and its arguments are unlimited.
        '''
        super(SpinTerm,self).__init__(id=id,mode='sp',value=value,modulate=modulate)
        self.neighbour=neighbour
        self.indexpacks=indexpacks
        self.amplitude=amplitude

    def __repr__(self):
        '''
        Convert an instance to string.
        '''
        result=[]
        result.append('id=%s'%self.id)
        result.append('value=%s'%self.value)
        result.append('neighbour=%s'%self.neighbour)
        result.append('indexpacks=%s'%self.indexpacks)
        if self.amplitude is not None:
            result.append('amplitude=%s'%self.amplitude)
        if hasattr(self,'modulate'):
            result.append('modulate=%s'%self.modulate)
        return 'SpinTerm('+', '.join(result)+')'

    def __add__(self,other):
        '''
        Overloaded operator(+), which supports the addition of a SpinTerm instance with a SpinTerm/SpinTerms instance.
        '''
        result=SpinTerms()
        result.append(self)
        if isinstance(other,SpinTerm):
            result.append(other)
        elif isinstance(other,SpinTerms):
            result.extend(other)
        else:
            raise ValueError('SpinTerm "+" error: the other parameter must be an instance of SpinTerm or SpinTerms.')
        return result

    def __pos__(self):
        '''
        Overloaded operator(+), i.e. +self.
        '''
        result=SpinTerms()
        result.append(self)
        return result

    def operators(self,bond,config,table=None,dtype=complex128):
        '''
        This method returns all the spin operators defined on the input bond with non-zero coefficients.

        Parameters
        ----------
        bond : Bond
            The bond on which the spin terms are defined.
        config : IDFConfig
            The configuration of spin degrees of freedom.
        table: Table, optional
            The index-sequence table.
        dtype: dtype,optional
            The data type of the coefficient of the returned operators.

        Returns
        -------
        Operators
            All the spin operators with non-zero coeffcients.
        '''
        result=Operators()
        epoint,spoint=bond.epoint,bond.spoint
        espin,sspin=config[epoint.pid],config[spoint.pid]
        if bond.neighbour==self.neighbour:
            value=self.value*(1 if self.amplitude==None else self.amplitude(bond))
            if abs(value)>RZERO:
                if callable(self.indexpacks):
                    buff=self.indexpacks(bond)
                else:
                    buff=self.indexpacks
                for obj in buff:
                    pv,tags,ms,orbitals=value*obj.value,obj.tags,obj.matrices,obj.orbitals
                    if len(tags)==1:
                        assert self.neighbour==0
                        for orbital in xrange(espin.norbital):
                            if orbitals[0] in (None,orbital):
                                eindex=Index(pid=epoint.pid,iid=SID(orbital=orbital,S=espin.S))
                                result+=SOperator(
                                    value=      pv,
                                    indices=    [eindex],
                                    spins=      [SpinMatrix(espin.S,tags[0],matrix=ms[0],dtype=dtype)],
                                    rcoords=    [epoint.rcoord],
                                    icoords=    [epoint.icoord],
                                    seqs=       None if table is None else (table[eindex])
                                    )
                    elif len(tags)==2:
                        for eorbital in xrange(espin.norbital):
                            if orbitals[0] in (None,eorbital):
                                for sorbital in xrange(sspin.norbital):
                                    if (orbitals[0] is not None and orbitals[1]==sorbital) or (orbitals[0] is None and sorbital==eorbital):
                                        eindex=Index(pid=epoint.pid,iid=SID(orbital=eorbital,S=espin.S))
                                        sindex=Index(pid=spoint.pid,iid=SID(orbital=sorbital,S=sspin.S))
                                        result+=SOperator(
                                            value=      pv,
                                            indices=    [eindex,sindex],
                                            spins=      [SpinMatrix(espin.S,tags[0],ms[0],dtype=dtype),SpinMatrix(sspin.S,tags[1],ms[1],dtype=dtype)],
                                            rcoords=    [epoint.rcoord,spoint.rcoord],
                                            icoords=    [epoint.icoord,spoint.icoord],
                                            seqs=       None if table is None else (table[eindex],table[sindex])
                                            )
                    else:
                        raise ValueError('SpinTerm operators error: not supported yet.')
        return result

class SpinTerms(Terms):
    '''
    This class packs several SpinTerm instances as a whole for convenience.
    '''

    def operators(self,bond,config,table,dtype=complex128):
        '''
        This method returns all the spin operators defined on the input bond with non-zero coefficients.

        Parameters
        ----------
        bond : Bond
            The bond on which the spin terms are defined.
        config : IDFConfig
            The configuration of spin degrees of freedom.
        table : Table, optional
            The index-sequence table.
        dtype : dtype,optional
            The data type of the coefficient of the returned operators.

        Returns
        -------
        Operators
            All the spin operators with non-zero coeffcients.
        '''
        result=Operators()
        for spinterm in self:
            result+=spinterm.operators(bond,config,table,dtype)
        return result
