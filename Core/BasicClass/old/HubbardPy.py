'''
Hubbard (onsite) interaction terms.
'''
from IndexPy import *
from OperatorPy import *
from TermPy import *
from BondPy import *
from TablePy import *

class Hubbard(Term):
    '''
    This class provides a complete description of single orbital and multi orbital Hubbard interactions.
    Attributes:
        value: float or 1D array-like, inherited from Term
            float: single-orbital Hubbard interaction.
            1D array-like: multi-orbital Hubbard interaction and the length must be 3.
                value[0]: intra-orbital interaction 
                value[1]: inter-orbital interaction
                value[2]: spin-flip and pair-hopping interaction
        atom: integer 
            The atom index of the point where the Hubbard interactions are defined.
    '''

    def __init__(self,tag,value,atom=None,modulate=None):
        '''
        Constructor.
        '''
        super(Hubbard,self).__init__('hb',tag,value,modulate)
        if atom is not None: self.atom=atom

    def __str__(self):
        '''
        Convert an instance to string.
        '''
        result=''
        if hasattr(self,'atom'): result+='Atom: '+str(self.atom)+'\n'
        result+='Tag,value: '+self.tag+','+str(self.value)+'\n'
        return result

    def __add__(self,other):
        '''
        Overloaded operator(+), which supports the addition of a Hubbard instance with a Hubbard/HubbardList instance.
        '''
        result=HubbardList(deepcopy(self))
        if isinstance(other,Hubbard):
            result.append(deepcopy(other))
        elif isinstance(other,HubbardList):
            result.extend(deepcopy(other))
        else:
            raise ValueError('Hubbard "+" error: the other parameter must be an instance of Hubbard or HubbardList')
        return result

    def __pos__(self):
        '''
        Overloaded operator(+), i.e. +self.
        '''
        result=HubbardList(deepcopy(self))
        return result

    def mesh(self,bond,dtype=float64):
        '''
        This method returns the mesh of Hubbard terms.
        '''
        ndim=bond.epoint.struct.norbital*bond.epoint.struct.nspin
        result=zeros((ndim,ndim,ndim,ndim),dtype=dtype)
        if hasattr(self,'atom'):
            atom=self.atom
        else:
            atom=bond.epoint.struct.atom
        if atom==bond.epoint.struct.atom:
            try:
                nv=len(self.value)
            except TypeError:
                nv=1
            if nv>=1:
                for h in xrange(bond.epoint.struct.norbital):
                    i=bond.epoint.struct.seq_state(h,1,ANNIHILATION)
                    j=bond.epoint.struct.seq_state(h,0,ANNIHILATION)
                    k=j
                    l=i
                    result[i,j,k,l]=self.value/2 if nv==1 else self.value[0]/2
            if nv==3:
                for h in xrange(bond.epoint.struct.norbital):
                    for g in xrange(bond.epoint.struct.norbital):
                      if g!=h:
                        i=bond.epoint.struct.seq_state(g,1,ANNIHILATION)
                        j=bond.epoint.struct.seq_state(h,0,ANNIHILATION)
                        k=j
                        l=i
                        result[i,j,k,l]=self.value[1]/2
                for h in xrange(bond.epoint.struct.norbital):
                    for g in xrange(h):
                        for f in xrange(2):
                            i=bond.epoint.struct.seq_state(g,f,ANNIHILATION)
                            j=bond.epoint.struct.seq_state(h,f,ANNIHILATION)
                            k=j
                            l=i
                            result[i,j,k,l]=(self.value[1]-self.value[2])/2
                for h in xrange(bond.epoint.struct.norbital):
                    for g in xrange(h):
                        i=bond.epoint.struct.seq_state(g,1,ANNIHILATION)
                        j=bond.epoint.struct.seq_state(h,0,ANNIHILATION)
                        k=bond.epoint.struct.seq_state(g,0,ANNIHILATION)
                        l=bond.epoint.struct.seq_state(h,1,ANNIHILATION)
                        result[i,j,k,l]=self.value[2]
                for h in xrange(bond.epoint.struct.norbital):
                    for g in xrange(h):
                        i=bond.epoint.struct.seq_state(g,1,ANNIHILATION)
                        j=bond.epoint.struct.seq_state(g,0,ANNIHILATION)
                        k=bond.epoint.struct.seq_state(h,0,ANNIHILATION)
                        l=bond.epoint.struct.seq_state(h,1,ANNIHILATION)
                        result[i,j,k,l]=self.value[2]
        return result

class HubbardList(list):
    '''
    This class pack several Hubbard instances as a whole for convenience.
    '''
    def __init__(self,*arg):
        self.mode='hb'
        for obj in arg:
            if isinstance(obj,Hubbard):
                self.append(obj)
            else:
                raise ValueError("HubbardList init error: the input argument should be Hubbard instances.")

    def __str__(self):
        '''
        Convert an instance to string.
        '''
        result='Hubbard terms:\n'
        for i,v in enumerate(self):
            result+='Node '+str(i)+':\n'+str(v)

    def __add__(self,other):
        '''
        Overloaded operator(+), which supports the addition of a HubbardList instance with a Hubbard/HubbardList instance.
        '''
        result=HubbardList(*deepcopy(self))
        if isinstance(other,Hubbard):
            result.append(deepcopy(other))
        elif isinstance(other,HubbardList):
            result.extend(deepcopy(other))
        else:
            raise ValueError('HubbardList "+" error: the other parameter must be an instance of Hubbard or HubbardList')
        return result

    def __mul__(self,other):
        '''
        Overloaded operator(*), which supports the left multiplication with a scalar.
        '''
        result=HubbardList()
        for obj in self:
            result.append(obj*other)
        return result

    def __rmul__(self,other):
        '''
        Overloaded operator(*), which supports the right multiplication with a scalar.
        '''
        return self.__mul__(other)

    def mesh(self,bond,dtype=float64):
        '''
        This method returns the mesh of all Hubbard terms defined on a bond.
        '''
        if bond.epoint.struct.nspin==2 and bond.spoint.struct.nspin==2:
            ndim=bond.epoint.struct.norbital*bond.epoint.struct.nspin
            result=zeros((ndim,ndim,ndim,ndim),dtype=dtype)
            if bond.neighbour==0:
                for obj in self:
                    result+=obj.mesh(bond,dtype=dtype)
            return result
        else:
            raise ValueError('HubbardList mesh error: the input bond must be onsite ones nspin=2.')

    def operators(self,bond,table,half=True,dtype=float64):
        '''
        This method returns all the Hubbard operators defined on the input bond with non-zero coefficients.
        Parameters:
            bond: Bond
                The bond on which the Hubbard terms is defined.
            table: Table
                The index-sequence table.
                Since Hubbard terms are quartic, it never uses the Nambu space.
            half: logical,optional
                When it is set to be True:
                1) only one half of the Hubbard operators is returned.
                2) the coefficient of the self-hermitian operators is also divided by a factor 2.
                The half==False case is not supported yet.
            dtype: dtype,optional
                The data type of the coefficient of the returned operators.
        Returns:
            result: OperatorList
                All the Hubbard operators with non-zero coeffcients.
        '''
        result=OperatorList()
        buff=self.mesh(bond,dtype=dtype)
        indices=argwhere(abs(buff)>RZERO)
        for (i,j,k,l) in indices:
            index1=Index(scope=bond.epoint.scope,site=bond.epoint.site,**bond.epoint.struct.state_index(i))
            index2=Index(scope=bond.epoint.scope,site=bond.epoint.site,**bond.epoint.struct.state_index(j))
            index3=Index(scope=bond.epoint.scope,site=bond.epoint.site,**bond.epoint.struct.state_index(k))
            index4=Index(scope=bond.epoint.scope,site=bond.epoint.site,**bond.epoint.struct.state_index(l))
            result.append(E_Hubbard(buff[i,j,k,l],indices=deepcopy([index1.dagger,index2.dagger,index3,index4]),rcoords=[bond.epoint.rcoord],icoords=[bond.epoint.icoord],seqs=[table[index1],table[index2],table[index3],table[index4]]))
        return result
