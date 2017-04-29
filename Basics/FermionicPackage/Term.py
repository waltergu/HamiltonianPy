'''
---------------
Fermionic terms
---------------

Fermionic terms, including:
    * classes: Quadratic, QuadracticList, Hubbard, HubbardList
    * functions: Hopping, Onsite, Pairing
'''

__all__=['Quadratic','QuadraticList','Hopping','Onsite','Pairing','Hubbard','HubbardList']

from numpy import *
from ..Constant import *
from ..Term import *
from ..DegreeOfFreedom import *
from ..Operator import * 
from DegreeOfFreedom import *
from Operator import * 

class Quadratic(Term):
    '''
    This class provides a complete and unified description for fermionic quadratic terms, i.e. hopping terms, onsite terms and pairing terms.

    Attributes
    ----------
    neighbour : integer
        The order of neighbour of this quadratic term.
    indexpacks : IndexPackList or function which returns IndexPackList
        The indexpacks of the quadratic term.
        When it is a function, it can return bond dependent indexpacks as needed.
    amplitude : function which returns float or complex
            This function can return bond dependent coefficient as needed.

    Notes
    -----
    The final coefficient comes from three parts, the value of itself, the value of the indexpacakge, and the value amplitude returns.
    '''
    
    def __init__(self,id,mode,value,neighbour=0,atoms=[],orbitals=[],spins=[],indexpacks=None,amplitude=None,modulate=None):
        '''
        Constructor.

        Parameters
        ----------
        id : string
            The specific id of the term.
        mode : string
            The type of the term.
        value : float or complex
            The overall coefficient of the term.
        neighbour : integer, optional
            The order of neighbour of the term.
        atoms,orbitals,spins : list of integer, optional
            The atom, orbital and spin index used to construct a wanted instance of FermiPack.
            When the parameter indexpacks is a function, these parameters will be omitted.
        indexpacks : IndexPackList or function, optional
            * IndexPackList:
                It will be multiplied by an instance of FermiPack constructed from atoms, orbitals and spins as the final indexpacks. 
            * function:
                It must return an instance of IndexPackList and take an instance of Bond as its only argument.
        amplitude: function, optional
            It must return a float or complex and take an instance of Bond as its only argument.
        modulate: function, optional
            It must return a float or complex and its arguments are unlimited.
        '''
        super(Quadratic,self).__init__(id=id,mode=mode,value=value,modulate=modulate)
        self.neighbour=neighbour
        if indexpacks is not None:
            if isinstance(indexpacks,IndexPackList):
                self.indexpacks=FermiPack(1,atoms=atoms,orbitals=orbitals,spins=spins)*indexpacks
            elif callable(indexpacks):
                self.indexpacks=indexpacks
            else:
                raise ValueError('Quadratic init error: the input indexpacks should be an instance of IndexPackList or a function.')
        else:
            self.indexpacks=IndexPackList(FermiPack(1,atoms=atoms,orbitals=orbitals,spins=spins))
        self.amplitude=amplitude

    def __repr__(self):
        '''
        Convert an instance to string.
        '''
        result=[]
        result.append('id=%s'%self.id)
        result.append('mode=%s'%self.mode)
        result.append('value=%s'%self.value)
        result.append('neighbour=%s'%self.neighbour)
        result.append('indexpacks=%s'%self.indexpacks)
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

    def mesh(self,bond,config,dtype=complex128):
        '''
        This method returns the mesh of a quadratic term defined on a bond.

        Parameters
        ----------
        bond : Bond
            The bond on which the quadratic term is defined.
        config : IDFConfig
            The configuration of the internal degrees of freedom.
        dtype : complex128,complex64,float64,float32, optional
            The data type of the returned mesh.

        Returns
        -------
        2D ndarray
            The mesh.
        '''
        edgr=config[bond.epoint.pid]
        sdgr=config[bond.spoint.pid]
        n1=edgr.norbital*edgr.nspin*edgr.nnambu
        n2=sdgr.norbital*sdgr.nspin*sdgr.nnambu
        result=zeros((n1,n2),dtype=dtype)
        if self.neighbour==bond.neighbour:
            value=self.value*(1 if self.amplitude==None else self.amplitude(bond))
            if callable(self.indexpacks):
                buff=self.indexpacks(bond)
            else:
                buff=self.indexpacks
            for obj in buff:
                pv=value*obj.value
                eatom=edgr.atom
                satom=sdgr.atom
                if hasattr(obj,'atoms'):
                    eatom=obj.atoms[0]
                    satom=obj.atoms[1]
                if eatom==edgr.atom and satom==sdgr.atom:
                    enambu=CREATION if self.mode=='pr' else ANNIHILATION
                    snambu=ANNIHILATION
                    if hasattr(obj,'spins'):
                        if hasattr(obj,'orbitals'):
                            i=edgr.seq_state(FID(obj.orbitals[0],obj.spins[0],enambu))
                            j=sdgr.seq_state(FID(obj.orbitals[1],obj.spins[1],snambu))
                            result[i,j]+=pv
                        elif edgr.norbital==sdgr.norbital:
                            for k in xrange(edgr.norbital):
                                i=edgr.seq_state(FID(k,obj.spins[0],enambu))
                                j=sdgr.seq_state(FID(k,obj.spins[1],snambu))
                                result[i,j]+=pv
                        else:
                            raise ValueError("Quadratic mesh error: the norbital on the epoint and the spoint of the input bond should be equal.")
                    elif edgr.nspin==sdgr.nspin:
                        if hasattr(obj,'orbitals'):
                            for k in xrange(edgr.nspin):
                                i=edgr.seq_state(FID(obj.orbitals[0],k,enambu))
                                j=sdgr.seq_state(FID(obj.orbitals[1],k,snambu))
                                result[i,j]+=pv
                        elif n1==n2:
                            ns=edgr.norbital*edgr.nspin
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
                    result[i,i]/=2.0
                    for j in xrange(i):
                        if abs(result[i,j]-conjugate(result[j,i]))<RZERO: result[i,j]=0
        return result

def Hopping(id,value,neighbour=1,atoms=[],orbitals=[],spins=[],indexpacks=None,amplitude=None,modulate=None):
    '''
    A specified function to construct a hopping term.
    '''
    return Quadratic(id,'hp',value,neighbour,atoms,orbitals,spins,indexpacks,amplitude,modulate)

def Onsite(id,value,neighbour=0,atoms=[],orbitals=[],spins=[],indexpacks=None,amplitude=None,modulate=None):
    '''
    A specified function to construct an onsite term.
    '''
    return Quadratic(id,'st',value,neighbour,atoms,orbitals,spins,indexpacks,amplitude,modulate)

def Pairing(id,value,neighbour=0,atoms=[],orbitals=[],spins=[],indexpacks=None,amplitude=None,modulate=None):
    '''
    A specified function to construct an pairing term.
    '''
    return Quadratic(id,'pr',value,neighbour,atoms,orbitals,spins,indexpacks,amplitude,modulate)

class QuadraticList(TermList):
    '''
    This class packs several Quadratic instances as a whole for convenience.
    '''

    def mesh(self,bond,config,select=None,dtype=complex128):
        '''
        This method returns the mesh of all quadratic terms defined on a bond.

        Parameters
        ----------
        bond : Bond
            The bond on which the quadratic terms are defined.
        config : IDFConfig
            The configuration of degrees of freedom.
        select : callable
            A function used to pick the quadratic terms whose only argument is an instance of Quadratic. 
            If the returned value if True, the selected quadratic term is included.
        dtype : complex128, complex64, float128, float64, optional
            The data type of the returned mesh.

        Returns
        -------
        2d ndarray
            The mesh.
        '''
        edgr,sdgr=config[bond.epoint.pid],config[bond.spoint.pid]
        if edgr.nnambu==sdgr.nnambu:
            result=zeros((edgr.norbital*edgr.nspin*edgr.nnambu,sdgr.norbital*sdgr.nspin*sdgr.nnambu),dtype=dtype)
            for obj in self:
                if select is None or select(obj):
                    result+=obj.mesh(bond,config,dtype=dtype)
            return result
        else:
            raise ValueError('Quadratic mesh error: the nnambu of epoint and spoint must be equal.')

    def operators(self,bond,config,table=None,dtype=complex128):
        '''
        This method returns all the desired quadratic operators defined on the input bond with non-zero coefficients.

        Parameters
        ----------
        bond : Bond
            The bond where the quadratic operators are defined.
        config : IDFConfig
            The configuration of degrees of freedom.
        table : Table, optional
            The index-sequence table.
            When it not None, only those operators with indices in it will be returned.
        dtype : dtype, optional
            The data type of the coefficient of the returned operators.

        Returns
        -------
        OperatorCollection
            The quadratic operators with non-zero coefficients.

        Notes
        -----
        Only one *half* of the operators are returned, which means
            * The Hermitian conjugate of non-Hermitian operators is not included;
            * The coefficient of the self-Hermitian operators is divided by a factor 2;
            * The BdG case, only the electron part of the hopping terms and onsite terms are contained, and for the electron part, the above rules also apply.
        '''
        result=to_operators(self.mesh(bond,config,dtype=dtype),bond,config,table)
        if bond.neighbour!=0:
            result+=to_operators(self.mesh(bond.reversed,config,select=lambda quadratic: True if quadratic.mode=='pr' else False,dtype=dtype),bond.reversed,config,table)
        return result

def to_operators(mesh,bond,config,table=None):
    result=OperatorCollection()
    indices=argwhere(abs(mesh)>RZERO)
    for (i,j) in indices:
        eindex=Index(bond.epoint.pid,config[bond.epoint.pid].state_index(i))
        sindex=Index(bond.spoint.pid,config[bond.spoint.pid].state_index(j))
        if table is None:
            result+=F_Quadratic(mesh[i,j],indices=(eindex.replace(nambu=1-eindex.nambu),sindex),rcoords=[bond.rcoord],icoords=[bond.icoord],seqs=None)
        elif eindex in table and sindex in table:
            result+=F_Quadratic(mesh[i,j],indices=(eindex.replace(nambu=1-eindex.nambu),sindex),rcoords=[bond.rcoord],icoords=[bond.icoord],seqs=(table[eindex],table[sindex]))
        else:
            etemp=eindex.replace(nambu=None)
            stemp=sindex.replace(nambu=None)
            if stemp in table and etemp in table:
                result+=F_Quadratic(mesh[i,j],indices=(eindex.replace(nambu=1-eindex.nambu),sindex),rcoords=[bond.rcoord],icoords=[bond.icoord],seqs=(table[etemp],table[stemp]))
    return result

class Hubbard(Term):
    '''
    This class provides a complete description of single orbital and multi orbital Hubbard interactions.

    Attributes
    ----------
    value : float or 1D array-like, inherited from Term
        * float:
            Single-orbital Hubbard interaction.
        * 1d array-like:
            Multi-orbital Hubbard interaction and the length must be 4.
                * value[0]: intra-orbital interaction
                * value[1]: inter-orbital interaction
                * value[2]: spin-flip interaction
                * value[3]: pair-hopping interaction
    atom : integer
        The atom index of the point where the Hubbard interactions are defined.
    '''

    def __init__(self,id,value,atom=None,modulate=None):
        '''
        Constructor.
        '''
        super(Hubbard,self).__init__(id=id,mode='hb',value=value,modulate=modulate)
        if atom is not None: self.atom=atom

    def __repr__(self):
        '''
        Convert an instance to string.
        '''
        result=[]
        result.append('id=%s'%self.id)
        if hasattr(self,'atom'): 
            result.append('atom=%s'%self.atom)
        result.append('value=%s'%self.value)
        return 'Hubbard('+', '.join(result)+')'

    def __add__(self,other):
        '''
        Overloaded operator(+), which supports the addition of a Hubbard instance with a Hubbard/HubbardList instance.
        '''
        result=HubbardList()
        result.append(self)
        if isinstance(other,Hubbard):
            result.append(other)
        elif isinstance(other,HubbardList):
            result.extend(other)
        else:
            raise ValueError('Hubbard "+" error: the other parameter must be an instance of Hubbard or HubbardList')
        return result

    def __pos__(self):
        '''
        Overloaded operator(+), i.e. +self.
        '''
        result=HubbardList()
        result.append(self)
        return result

    def mesh(self,bond,config,dtype=float64):
        '''
        This method returns the mesh of Hubbard terms.

        Parameters
        ----------
        bond : Bond
            The bond on which the Hubbard term is defined.
        config : IDFConfig
            The configuration of internal degrees of freedom.
        dtype : complex128,complex64,float128,float64, optional
            The data type of the returned mesh.

        Returns
        -------
        4d ndarray
            The mesh.
        '''
        dgr=config[bond.epoint.pid]
        ndim=dgr.norbital*dgr.nspin
        result=zeros((ndim,ndim,ndim,ndim),dtype=dtype)
        if hasattr(self,'atom'):
            atom=self.atom
        else:
            atom=dgr.atom
        if atom==dgr.atom:
            try:
                nv=len(self.value)
            except TypeError:
                nv=1
            if nv>=1:
                for h in xrange(dgr.norbital):
                    i=dgr.seq_state(FID(h,1,ANNIHILATION))
                    j=dgr.seq_state(FID(h,0,ANNIHILATION))
                    k=j
                    l=i
                    result[i,j,k,l]=self.value/2 if nv==1 else self.value[0]/2
            if nv==4:
                for h in xrange(dgr.norbital):
                    for g in xrange(dgr.norbital):
                      if g!=h:
                        i=dgr.seq_state(FID(g,1,ANNIHILATION))
                        j=dgr.seq_state(FID(h,0,ANNIHILATION))
                        k=j
                        l=i
                        result[i,j,k,l]=self.value[1]/2
                for h in xrange(dgr.norbital):
                    for g in xrange(h):
                        for f in xrange(2):
                            i=dgr.seq_state(FID(g,f,ANNIHILATION))
                            j=dgr.seq_state(FID(h,f,ANNIHILATION))
                            k=j
                            l=i
                            result[i,j,k,l]=(self.value[1]-self.value[2])/2
                for h in xrange(dgr.norbital):
                    for g in xrange(h):
                        i=dgr.seq_state(FID(g,1,ANNIHILATION))
                        j=dgr.seq_state(FID(h,0,ANNIHILATION))
                        k=dgr.seq_state(FID(g,0,ANNIHILATION))
                        l=dgr.seq_state(FID(h,1,ANNIHILATION))
                        result[i,j,k,l]=self.value[2]
                for h in xrange(dgr.norbital):
                    for g in xrange(h):
                        i=dgr.seq_state(FID(g,1,ANNIHILATION))
                        j=dgr.seq_state(FID(g,0,ANNIHILATION))
                        k=dgr.seq_state(FID(h,0,ANNIHILATION))
                        l=dgr.seq_state(FID(h,1,ANNIHILATION))
                        result[i,j,k,l]=self.value[3]
        return result

class HubbardList(TermList):
    '''
    This class pack several Hubbard instances as a whole for convenience.
    '''

    def mesh(self,bond,config,dtype=float64):
        '''
        This method returns the mesh of all Hubbard terms defined on a bond.

        Parameters
        ----------
        bond : Bond
            The bond on which the Hubbard term is defined.
        config : Configuration
            The configuration of degrees of freedom.
        dtype : complex128,complex64,float128,float64, optional
            The data type of the returned mesh.

        Returns
        -------
        4d ndarray
            The mesh.
        '''
        if bond.neighbour==0:
            edgr,sdgr=config[bond.epoint.pid],config[bond.spoint.pid]
            if edgr.nspin==2 and sdgr.nspin==2:
                ndim=edgr.norbital*edgr.nspin
                result=zeros((ndim,ndim,ndim,ndim),dtype=dtype)
                for obj in self:
                    result+=obj.mesh(bond,config,dtype=dtype)
                return result
            else:
                raise ValueError('HubbardList mesh error: the input bond must be onsite ones nspin=2.')
        else:
            return 0

    def operators(self,bond,config,table=None,dtype=float64):
        '''
        This method returns all the Hubbard operators defined on the input bond with non-zero coefficients.

        Parameters
        ----------
        bond : Bond
            The bond on which the Hubbard terms are defined.
        config: IDFConfig
            The configuration of internal degrees of freedom.
        table: Table, optional
            The index-sequence table. Since Hubbard terms are quartic, it never uses the Nambu space.
        dtype: dtype,optional
            The data type of the coefficient of the returned operators.
 
        Returns
        -------
        OperatorCollection
            All the Hubbard operators with non-zero coeffcients.

        Notes
        -----
        Only one "half" of the operators are returned, which means
            * The Hermitian conjugate of non-Hermitian operators is not included;
            * The coefficient of the self-Hermitian operators is divided by a factor 2.
        '''
        result=OperatorCollection()
        dgr=config[bond.epoint.pid]
        mesh=self.mesh(bond,config,dtype=dtype)
        indices=argwhere(abs(mesh)>RZERO)
        for (i,j,k,l) in indices:
            index1=Index(bond.epoint.pid,dgr.state_index(i))
            index2=Index(bond.epoint.pid,dgr.state_index(j))
            index3=Index(bond.epoint.pid,dgr.state_index(k))
            index4=Index(bond.epoint.pid,dgr.state_index(l))
            if table is None:
                seqs=None
            elif index1 in table and index2 in table and index3 in table and index4 in table:
                seqs=(table[index1],table[index2],table[index3],table[index4])
            else:
                seqs=(table[index1.mask('nambu')],table[index2.mask('nambu')],table[index3.mask('nambu')],table[index4.mask('nambu')])
            result+=F_Hubbard(
                value=      mesh[i,j,k,l],
                indices=    (index1.replace(nambu=CREATION),index2.replace(nambu=CREATION),index3,index4),
                rcoords=    [bond.epoint.rcoord],
                icoords=    [bond.epoint.icoord],
                seqs=       seqs
                )
        return result
