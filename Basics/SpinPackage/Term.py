'''
----------
Spin terms
----------

Spin terms, including:
    * classes: SpinTerm
'''

__all__=['SpinTerm']

from ..Utilities import RZERO,decimaltostr
from ..Term import *
from ..DegreeOfFreedom import *
from ..Operator import * 
from DegreeOfFreedom import *
from Operator import *
import numpy as np

class SpinTerm(Term):
    '''
    This class provides a complete and unified description for spin terms.

    Attributes
    ----------
    neighbour : int
        The order of neighbour of this spin term.
    indexpacks : IndexPacks or function which returns IndexPacks
        The indexpacks of the spin term.
        When it is a function, it returns bond dependent indexpacks as needed.
    amplitude : function which returns float or complex
        This function returns bond dependent coefficient as needed.
    '''

    def __init__(self,id,value=1.0,neighbour=1,indexpacks=None,amplitude=None,modulate=None):
        '''
        Constructor.

        Parameters
        ----------
        id : str
            The specific id of the term.
        value : float or complex
            The overall coefficient of the term.
        neighbour : int
            The order of neighbour of the term.
        indexpacks : IndexPacks or callable
            * IndexPacks:
                The indexpacks of the term.
            * callable in the form ``indexpacks(bond)``:
                It returns the bond-dependent indexpacks of the term.
        amplitude: callable in the form ``amplitude(bond)``, optional
            It returns the bond-dependent amplitude of the term.
        modulate: callable in the form ``modulate(*arg,**karg)``, optional
            This function defines the way to change the overall coefficient of the term dynamically.
        '''
        super(SpinTerm,self).__init__(id=id,value=value,modulate=modulate)
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
        if self.amplitude is not None: result.append('amplitude=%s'%self.amplitude)
        if self.modulate is not None: result.append('modulate=%s'%self.modulate)
        return 'SpinTerm('+', '.join(result)+')'

    def operators(self,bond,config,table=None,dtype=np.complex128,**karg):
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
        dtype : np.complex128, np.float64, optional
            The data type of the coefficient of the returned operators.

        Returns
        -------
        Operators
            All the spin operators with non-zero coefficients.
        '''
        result=Operators()
        epoint,spoint=bond.epoint,bond.spoint
        espin,sspin=config[epoint.pid],config[spoint.pid]
        if bond.neighbour==self.neighbour:
            value=self.value*(1 if self.amplitude is None else self.amplitude(bond))
            if abs(value)>RZERO:
                for spack in self.indexpacks(bond) if callable(self.indexpacks) else self.indexpacks:
                    pv,tags,ms,orbitals=value*spack.value,spack.tags,spack.matrices,spack.orbitals
                    if len(tags)==1:
                        assert self.neighbour==0
                        for orbital in xrange(espin.norbital):
                            if orbitals[0] in (None,orbital):
                                eindex=Index(pid=epoint.pid,iid=SID(orbital=orbital,S=espin.S))
                                result+=SOperator(
                                    value=      pv,
                                    indices=    [eindex],
                                    spins=      [SpinMatrix(espin.S,tags[0],matrix=ms[0],dtype=dtype)],
                                    seqs=       None if table is None else (table[eindex],)
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
                                            seqs=       None if table is None else (table[eindex],table[sindex])
                                            )
                    else:
                        raise ValueError('SpinTerm operators error: not supported yet.')
        return result

    @property
    def unit(self):
        '''
        The unit term.
        '''
        return self.replace(value=1.0)

    def strrep(self,bond,config):
        '''
        The string representation of the term on a bond.

        Parameters
        ----------
        bond : Bond
            The bond where the term is to be represented.
        config : IDFConfig
            The configuration of internal degrees of freedom.

        Returns
        -------
        str
            The string representation of the term on the bond.
        '''
        result=[]
        if self.neighbour==bond.neighbour:
            value=self.value*(1 if self.amplitude is None else self.amplitude(bond))
            if np.abs(value)>RZERO:
                for spack in self.indexpacks(bond) if callable(self.indexpacks) else self.indexpacks:
                    result.append('sp:%s*%s'%(decimaltostr(value,Term.NDECIMAL),repr(spack)))
        return '\n'.join(result)
