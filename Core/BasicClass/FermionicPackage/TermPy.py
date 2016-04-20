'''
Fermionic terms, including:
1) classes: Quadratic, QuadracticList, Hubbard, HubbardList
2) functions: Hopping, Onsite, Pairing
'''

#from ..Geometry import *
from ..TermPy import *
#from ..DegreeOfFreedomPy import *
#from DegreeOfFreedomPy import *
#from OperatorPy import * 

__all__=['Quadratic','QuadraticList','Hopping','Onsite','Pairing','Hubbard','HubbardList']

class Quadratic(Term):
    '''
    This class provides a complete and unified description for fermionic quadratic terms, i.e. hopping terms, onsite terms and pairing terms.
    Attributes:
        neighbour: integer
            The order of neighbour of this quadratic term.
        indexpackages: IndexPackageList or function which returns IndexPackageList
            The indexpackages of the quadratic term.
            When it is a function, it can return bond dependent indexpackages as needed.
        amplitude: function which returns float or complex
            This function can return bond dependent and index dependent coefficient as needed.
    Note: The final coefficient comes from three parts, the value of itself, the value of the indexpacakge, and the value amplitude returns.
    '''
    
    def __init__(self,id,mode,value,neighbour=0,atoms=[],orbitals=[],spins=[],indexpackages=None,amplitude=None,modulate=None):
        '''
        Constructor.
        Parameters:
            id: string
                The specific id of the term.
            mode: string
                The type of the term.
            value: float or complex
                The overall coefficient of the term.
            neighbour: integer, optional
                The order of neighbour of the term.
            atoms,orbitals,spins: list of integer, optional
                The atom, orbital and spin index used to construct a wanted instance of IndexPackage.
                When the parameter indexpackages is a function, these parameters will be omitted.
            indexpackages: IndexPackageList or function
                1) IndexPackageList:
                    It will be multiplied by an instance of IndexPackage constructed from atoms, orbitals and spins as the final indexpackages. 
                2) function:
                    It must return an instance of IndexPackageList and take an instance of Bond as its only argument.
            amplitude: function
                It must return a float or complex and take an instance of Bond as its only argument.
            modulate: function
                It must return a float or complex and its arguments are unlimited.
        '''
        super(Quadratic,self).__init__(id,mode,value,modulate)
        self.neighbour=neighbour
        if indexpackages is not None:
            if isinstance(indexpackages,IndexPackageList):
                self.indexpackages=IndexPackage(1,atoms=atoms,orbitals=orbitals,spins=spins)*indexpackages
            elif callable(indexpackages):
                self.indexpackages=indexpackages
            else:
                raise ValueError('Quadratic init error: the input indexpackages should be an instance of IndexPackageList or a function.')
        else:
            self.indexpackages=IndexPackageList(IndexPackage(1,atoms=atoms,orbitals=orbitals,spins=spins))
        self.amplitude=amplitude

    def __repr__(self):
        '''
        Convert an instance to string.
        '''
        result=[]
        result.append('id=%s'%self.id)
        result.append('mode=%s'%self.mode)
        result.append('value=%s'%self.value)
        result.append('indexpackages=%s'%self.indexpackages)
        if self.amplitude is not None:
            result.append('amplitude=%s'%self.amplitude)
        if hasattr(self,'modulate'):
            result.append('modulate=%s'%self.modulate)
        return 'Quadratic('+', '.join(result)+')'

    def __add__(self,other):
        '''
        Overloaded operator(+), which supports the addition of a Quadratic instance with a Quadratic/QuadraticList instance.
        '''
        result=QuadraticList()
        result.append(self)
        if isinstance(other,Quadratic):
            result.append(other)
        elif isinstance(other,QuadraticList):
            result.extend(other)
        else:
            raise ValueError('Quadratic "+" error: the other parameter must be an instance of Quadratic or QuadraticList.')
        return result

    def __pos__(self):
        '''
        Overloaded operator(+), i.e. +self.
        '''
        result=QuadraticList()
        result.append(self)
        return result

    def mesh(self,bond,half=True,dtype=complex128):
        '''
        This method returns the mesh of a quadratic term defined on a bond.
        '''
        n1=bond.epoint.struct.norbital*bond.epoint.struct.nspin*bond.epoint.struct.nnambu
        n2=bond.spoint.struct.norbital*bond.spoint.struct.nspin*bond.spoint.struct.nnambu
        result=zeros((n1,n2),dtype=dtype)
        if self.neighbour==bond.neighbour:
            value=self.value*(1 if self.amplitude==None else self.amplitude(bond))
            if callable(self.indexpackages):
                buff=self.indexpackages(bond)
            else:
                buff=self.indexpackages
            for obj in buff:
                pv=value*obj.value
                eatom=bond.epoint.struct.atom
                satom=bond.spoint.struct.atom
                if hasattr(obj,'atoms'):
                    eatom=obj.atoms[0]
                    satom=obj.atoms[1]
                if eatom==bond.epoint.struct.atom and satom==bond.spoint.struct.atom:
                    enambu=CREATION if self.mode=='pr' else ANNIHILATION
                    snambu=ANNIHILATION
                    if hasattr(obj,'spins'):
                        if hasattr(obj,'orbitals'):
                            i=bond.epoint.struct.seq_state(obj.orbitals[0],obj.spins[0],enambu)
                            j=bond.spoint.struct.seq_state(obj.orbitals[1],obj.spins[1],snambu)
                            result[i,j]+=pv
                        elif bond.epoint.struct.norbital==bond.spoint.struct.norbital:
                            for k in xrange(bond.epoint.struct.norbital):
                                i=bond.epoint.struct.seq_state(k,obj.spins[0],enambu)
                                j=bond.spoint.struct.seq_state(k,obj.spins[1],snambu)
                                result[i,j]+=pv
                        else:
                            raise ValueError("Quadratic mesh error: the norbital of epoint and spoint of the input bond should be equal.")
                    elif bond.epoint.struct.nspin==bond.spoint.struct.nspin:
                        if hasattr(obj,'orbitals'):
                            for k in xrange(bond.epoint.struct.nspin):
                                i=bond.epoint.struct.seq_state(obj.orbitals[0],k,enambu)
                                j=bond.spoint.struct.seq_state(obj.orbitals[1],k,snambu)
                                result[i,j]+=pv
                        elif n1==n2:
                            ns=bond.epoint.struct.norbital*bond.epoint.struct.nspin
                            if self.mode=='pr':
                                for k in xrange(ns):
                                    result[k,k+ns]+=pv
                            else:
                                for k in xrange(ns):
                                    result[k,k]+=pv
                        else:
                            raise ValueError("Quadratic mesh error: the norbital of epoint and spoint of the input bond should be equal.")
                    else:
                        raise ValueError("Quadratic mesh error: the nspin of epoint and spoint of the input bond should be equal.")
            if self.neighbour==0:
                for i in xrange(n1):
                    if half: result[i,i]/=2
                    for j in xrange(i):
                        if abs(result[i,j]-conjugate(result[j,i]))<RZERO: result[i,j]=0
        return result

def Hopping(id,value,neighbour=1,atoms=[],orbitals=[],spins=[],indexpackages=None,amplitude=None,modulate=None):
    '''
    A specified function to construct a hopping term.
    '''
    return Quadratic(id,'hp',value,neighbour,atoms,orbitals,spins,indexpackages,amplitude,modulate)

def Onsite(id,value,neighbour=0,atoms=[],orbitals=[],spins=[],indexpackages=None,amplitude=None,modulate=None):
    '''
    A specified function to construct an onsite term.
    '''
    return Quadratic(id,'st',value,neighbour,atoms,orbitals,spins,indexpackages,amplitude,modulate)

def Pairing(id,value,neighbour=0,atoms=[],orbitals=[],spins=[],indexpackages=None,amplitude=None,modulate=None):
    '''
    A specified function to construct an pairing term.
    '''
    return Quadratic(id,'pr',value,neighbour,atoms,orbitals,spins,indexpackages,amplitude,modulate)

class QuadraticList(list):
    '''
    This class packs several Quadratic instances as a whole for convenience.
    '''
    
    def __init__(self,*arg):
        for obj in arg:
            if isinstance(obj,Quadratic):
                self.append(obj)
            else:
                raise ValueError("QuadraticList init error: the input argument should be Quadratic instances.")

    def __str__(self):
        '''
        Convert an instance to string.
        '''
        return '\n'.join(['Node[%s]:%s'%(i,obj) for i,obj in enumerate(self)])

    def __add__(self,other):
        '''
        Overloaded operator(+), which supports the addition of a QuadraticList instance with a Quadratic/QuadraticList instance.
        '''
        result=copy(self)
        if isinstance(other,Quadratic):
            result.append(other)
        elif isinstance(other,QuadraticList):
            result.extend(other)
        else:
            raise ValueError('QuadraticList "+" error: the other parameter must be an instance of Quadratic or QuadraticList.')
        return result

    def __mul__(self,other):
        '''
        Overloaded operator(*), which supports the left multiplication with a scalar.
        '''
        result=QuadraticList()
        for obj in self:
            result.append(obj*other)
        return result

    def __rmul__(self,other):
        '''
        Overloaded operator(*), which supports the right multiplication with a scalar.
        '''
        return self.__mul__(other)

    def mesh(self,bond,half,mask=None,dtype=complex128):
        '''
        This method returns the mesh of all quadratic terms defined on a bond.
        '''
        if bond.epoint.struct.nnambu==bond.spoint.struct.nnambu:
            n1=bond.epoint.struct.norbital*bond.epoint.struct.nspin*bond.epoint.struct.nnambu
            n2=bond.spoint.struct.norbital*bond.spoint.struct.nspin*bond.spoint.struct.nnambu
            result=zeros((n1,n2),dtype=dtype)
            for obj in self:
                if mask is None or mask(obj):
                    result+=obj.mesh(bond,half,dtype=dtype)
            return result
        else:
            raise ValueError('Quadratic mesh error: the nnambu of epoint and spoint must be equal.')

    def operators(self,bond,table,half=True,dtype=complex128):
        '''
        This method returns all the desired quadratic operators defined on the input bond with non-zero coefficients.
        Parameters:
            bond: Bond
                The bond where the quadratic operators are defined.
            table: Table
                The index-sequence table.
                Only those operators whose indices are in this table will be returned.
            half: logical,optional
                When it is set to be True:
                1) only one half of the quadratic operators is returned.
                2) the coefficient of the self-hermitian operators is also divided by a factor 2.
                3) as for the BdG case, only the electron part of the hopping terms and onsite terms are contained.
            dtype: dtype, optional
                The data type of the coefficient of the returned operators.
        Returns:
            result: OperatorList
                The quadratic operators with non-zero coefficients.
        '''
        result=_operators(self.mesh(bond,half,dtype=dtype),bond,table,half)
        if bond.neighbour!=0:
            result.extend(_operators(self.mesh(bond.reversed,half,mask=lambda quadratic: True if quadratic.mode=='pr' else False,dtype=dtype),bond.reversed,table,half))
        return result

def _operators(mesh,bond,table,half=True):
    result=OperatorList()
    indices=argwhere(abs(mesh)>RZERO)
    for (i,j) in indices:
        eindex=Index(scope=bond.epoint.scope,site=bond.epoint.site,**bond.epoint.struct.state_index(i))
        sindex=Index(scope=bond.spoint.scope,site=bond.spoint.site,**bond.epoint.struct.state_index(j))
        if eindex in table and sindex in table:
            result.append(E_Quadratic(mesh[i,j],indices=deepcopy([eindex.dagger,sindex]),rcoords=[bond.rcoord],icoords=[bond.icoord],seqs=[table[eindex],table[sindex]]))
            if not half and eindex!=sindex:
                result.append(E_Quadratic(conjugate(mesh[i,j]),indices=deepcopy([sindex.dagger,eindex]),rcoords=[-bond.rcoord],icoords=[-bond.icoord],seqs=[table[sindex],table[eindex]]))
    return result

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

    def __init__(self,id,value,atom=None,modulate=None):
        '''
        Constructor.
        '''
        super(Hubbard,self).__init__('hb',id,value,modulate)
        if atom is not None: self.atom=atom

    def __str__(self):
        '''
        Convert an instance to string.
        '''
        result=''
        if hasattr(self,'atom'): result+='Atom: '+str(self.atom)+'\n'
        result+='id,value: '+self.id+','+str(self.value)+'\n'
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
