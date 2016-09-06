'''
Spin terms, including:
1) classes: SpinTerm, SpinTermList
'''

__all__=['SpinTerm','SpinTermList']

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
    Attributes:
        neighbour: integer
            The order of neighbour of this spin term.
        indexpacks: IndexPackList or function which returns IndexPackList
            The indexpacks of the spin term.
            When it is a function, it can return bond dependent indexpacks as needed.
        amplitude: function which returns float or complex
            This function can return bond dependent coefficient as needed.
    '''

    def __init__(self,id,value,neighbour,indexpacks,amplitude=None,modulate=None):
        '''
        Constructor.
        Parameters:
            id: string
                The specific id of the term.
            value: float or complex
                The overall coefficient of the term.
            neighbour: integer
                The order of neighbour of the term.
            indexpacks: IndexPackList or function
                When it is a function, it must return an instance of IndexPackList and take an instance of Bond as its only argument.
            amplitude: function, optional
                It must return a float or complex and take an instance of Bond as its only argument.
            modulate: function, optional
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
        Overloaded operator(+), which supports the addition of a SpinTerm instance with a SpinTerm/SpinTermList instance.
        '''
        result=SpinTermList()
        result.append(self)
        if isinstance(other,SpinTerm):
            result.append(other)
        elif isinstance(other,SpinTermList):
            result.extend(other)
        else:
            raise ValueError('SpinTerm "+" error: the other parameter must be an instance of SpinTerm or SpinTermList.')
        return result

    def __pos__(self):
        '''
        Overloaded operator(+), i.e. +self.
        '''
        result=SpinTermList()
        result.append(self)
        return result

    def operators(self,bond,config,table=None,dtype=complex128):
        '''
        This method returns all the spin operators defined on the input bond with non-zero coefficients.
        Parameters:
            bond: Bond
                The bond on which the spin terms are defined.
            config: Configuration
                The configuration of spin degrees of freedom.
            table: Table, optional
                The index-sequence table.
            dtype: dtype,optional
                The data type of the coefficient of the returned operators.
        Returns:
            result: OperatorCollection
                All the spin operators with non-zero coeffcients.
        '''
        result=OperatorCollection()
        eS=config[bond.epoint.pid].S
        sS=config[bond.spoint.pid].S
        eindex=Index(pid=bond.epoint.pid,iid=SID(S=eS))
        sindex=Index(pid=bond.spoint.pid,iid=SID(S=sS))
        if bond.neighbour==self.neighbour:
            value=self.value*(1 if self.amplitude==None else self.amplitude(bond))
            if abs(value)>RZERO:
                if callable(self.indexpacks):
                    buff=self.indexpacks(bond)
                else:
                    buff=self.indexpacks
                for obj in buff:
                    pv=value*obj.value
                    pack=obj.pack
                    if self.neighbour==0:
                        if len(pack)!=1:
                            raise ValueError('SpinTerm operators error: the length of the pack of each SpinPack must be 1 when neighbour is 0.')
                        id,m=pack[0] if hasattr(pack[0],'__iter__') else (pack[0],None)
                        result+=OperatorS(
                            value=      pv,
                            indices=    [eindex],
                            spins=      [SpinMatrix((eS,id),matrix=m,dtype=dtype)],
                            rcoords=    [bond.epoint.rcoord],
                            icoords=    [bond.epoint.icoord],
                            seqs=       None if table is None else (table[eindex])
                            )
                    else:
                        if len(pack)!=2:
                            raise ValueError('SpinTerm operators error: the length of the pack of each SpinPack must be 2 if neighbour is not 0.')
                        id0,m0=pack[0] if hasattr(pack[0],'__iter__') else (pack[0],None)
                        id1,m1=pack[1] if hasattr(pack[1],'__iter__') else (pack[1],None)
                        result+=OperatorS(
                            value=      pv,
                            indices=    [eindex,sindex],
                            spins=      [SpinMatrix((eS,id0),matrix=m0,dtype=dtype),SpinMatrix((sS,id1),matrix=m1,dtype=dtype)],
                            rcoords=    [bond.epoint.rcoord,bond.spoint.rcoord],
                            icoords=    [bond.epoint.icoord,bond.spoint.icoord],
                            seqs=       None if table is None else (table[eindex],table[sindex])
                            )
        return result

class SpinTermList(TermList):
    '''
    This class packs several SpinTerm instances as a whole for convenience.
    '''

    def operators(self,bond,config,table,dtype=complex128):
        '''
        This method returns all the spin operators defined on the input bond with non-zero coefficients.
        Parameters:
            bond: Bond
                The bond on which the spin terms are defined.
            config: Configuration
                The configuration of spin degrees of freedom.
            table: Table, optional
                The index-sequence table.
            dtype: dtype,optional
                The data type of the coefficient of the returned operators.
        Returns:
            result: OperatorCollection
                All the spin operators with non-zero coeffcients.
        '''
        result=OperatorCollection()
        for spinterm in self:
            result+=spinterm.operators(bond,config,table,dtype)
        return result
