'''
---------------
Fermionic terms
---------------

Fermionic terms, including:
    * classes: Quadratic, Hubbard
    * functions: Hopping, Onsite, Pairing
'''

__all__=['Quadratic','Hopping','Onsite','Pairing','Hubbard']

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
    indexpacks : IndexPacks or function which returns IndexPacks
        The indexpacks of the quadratic term.
        When it is a function, it can return bond dependent indexpacks as needed.
    amplitude : function which returns float or complex
            This function can return bond dependent coefficient as needed.

    Notes
    -----
    The final coefficient comes from three parts, the value of itself, the value of the indexpack, and the value amplitude returns.
    '''
    
    def __init__(self,id,mode,value,neighbour=0,atoms=(),orbitals=(),spins=(),indexpacks=None,amplitude=None,modulate=None):
        '''
        Constructor.

        Parameters
        ----------
        id : string
            The specific id of the term.
        mode : 'hp','st','pr'
            The type of the term.
        value : float or complex
            The overall coefficient of the term.
        neighbour : integer, optional
            The order of neighbour of the term.
        atoms,orbitals,spins : list of integer, optional
            The atom, orbital and spin index used to construct a wanted instance of FermiPack.
            When the parameter indexpacks is a function, these parameters will be omitted.
        indexpacks : IndexPacks or function, optional
            * IndexPacks:
                It will be multiplied by an instance of FermiPack constructed from atoms, orbitals and spins as the final indexpacks. 
            * function:
                It must return an instance of IndexPacks and take an instance of Bond as its only argument.
        amplitude: function, optional
            It must return a float or complex and take an instance of Bond as its only argument.
        modulate: function, optional
            It must return a float or complex and its arguments are unlimited.
        '''
        assert mode in ('hp','st','pr')
        super(Quadratic,self).__init__(id=id,mode=mode,value=value,modulate=modulate)
        self.neighbour=neighbour
        if indexpacks is None:
            self.indexpacks=IndexPacks(FermiPack(1,atoms=atoms,orbitals=orbitals,spins=spins))
        else:
            if isinstance(indexpacks,IndexPacks):
                self.indexpacks=FermiPack(1,atoms=atoms,orbitals=orbitals,spins=spins)*indexpacks
            elif callable(indexpacks):
                self.indexpacks=lambda bond: FermiPack(1,atoms=atoms,orbitals=orbitals,spins=spins)*indexpacks(bond)
            else:
                raise ValueError('Quadratic init error: the input indexpacks should be an instance of IndexPacks or a function.')
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
        if self.modulate is not None:
            result.append('modulate=%s'%self.modulate)
        return 'Quadratic('+', '.join(result)+')'

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
            value=self.value*(1 if self.amplitude is None else self.amplitude(bond))
            for obj in self.indexpacks(bond) if callable(self.indexpacks) else self.indexpacks:
                pv=value*obj.value
                if not hasattr(obj,'atoms') or (edgr.atom,sdgr.atom)==obj.atoms:
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
                            dn=edgr.norbital*edgr.nspin if self.mode=='pr' else 0
                            for k in xrange(edgr.norbital*edgr.nspin):
                                result[k,k+dn]+=pv
                        else:
                            raise ValueError("Quadratic mesh error: the norbital of epoint and spoint of the input bond should be equal.")
                    else:
                        raise ValueError("Quadratic mesh error: the nspin of epoint and spoint of the input bond should be equal.")
        return result

    def operators(self,bond,config,table=None,half=True,dtype=complex128,**karg):
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
        half : logical, optional
            When True, only one half of the operators are returned, which means
                * The Hermitian conjugate of non-Hermitian operators is not included;
                * The coefficient of the self-Hermitian operators is divided by a factor 2.
        dtype : dtype, optional
            The data type of the coefficient of the returned operators.

        Returns
        -------
        Operators
            The quadratic operators with non-zero coefficients.

        Notes
        -----
        No matter whether or not ``half`` is True, for the BdG case, only the electron part of the hopping terms and onsite terms are contained.
        '''
        def _operators_(mesh,bond,config,table=None):
            result=Operators()
            indices=argwhere(abs(mesh)>RZERO)
            for (i,j) in indices:
                eindex=Index(bond.epoint.pid,config[bond.epoint.pid].state_index(i))
                sindex=Index(bond.spoint.pid,config[bond.spoint.pid].state_index(j))
                if table is None:
                    result+=FQuadratic(
                            value=      mesh[i,j],
                            indices=    (eindex.replace(nambu=1-eindex.nambu),sindex),
                            seqs=       None,
                            rcoord=     bond.rcoord,
                            icoord=     bond.icoord
                            )
                else:
                    masks=next(iter(table)).masks
                    etemp=eindex.mask(*masks)
                    stemp=sindex.mask(*masks)
                    if stemp in table and etemp in table:
                        result+=FQuadratic(
                            value=      mesh[i,j],
                            indices=    (eindex.replace(nambu=1-eindex.nambu),sindex),
                            seqs=       (table[etemp],table[stemp]),
                            rcoord=     bond.rcoord,
                            icoord=     bond.icoord
                            )
            return result
        if self.mode=='st':
            mesh=self.mesh(bond,config,dtype=dtype)
            if half:
                for i in xrange(mesh.shape[0]):
                    mesh[i,i]/=2.0
                    for j in xrange(i):
                        if abs(mesh[i,j]-conjugate(mesh[j,i]))<RZERO: mesh[i,j]=0
            result=_operators_(mesh,bond,config,table)
        else:
            result=_operators_(self.mesh(bond,config,dtype=dtype),bond,config,table)
            if self.mode=='pr' and bond.neighbour!=0: result+=_operators_(self.mesh(bond.reversed,config,dtype=dtype),bond.reversed,config,table)
            if not half: result+=result.dagger
        return result

def Hopping(id,value,neighbour=1,atoms=(),orbitals=(),spins=(),indexpacks=None,amplitude=None,modulate=None):
    '''
    A specified function to construct a hopping term.
    '''
    return Quadratic(id,'hp',value,neighbour,atoms,orbitals,spins,indexpacks,amplitude,modulate)

def Onsite(id,value,atoms=(),orbitals=(),spins=(),indexpacks=None,amplitude=None,modulate=None):
    '''
    A specified function to construct an onsite term.
    '''
    return Quadratic(id,'st',value,0,atoms,orbitals,spins,indexpacks,amplitude,modulate)

def Pairing(id,value,neighbour=0,atoms=(),orbitals=(),spins=(),indexpacks=None,amplitude=None,modulate=None):
    '''
    A specified function to construct an pairing term.
    '''
    return Quadratic(id,'pr',value,neighbour,atoms,orbitals,spins,indexpacks,amplitude,modulate)

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
        self.atom=atom

    def __repr__(self):
        '''
        Convert an instance to string.
        '''
        result=[]
        result.append('id=%s'%self.id)
        if self.atom is not None: result.append('atom=%s'%self.atom)
        result.append('value=%s'%self.value)
        return 'Hubbard('+', '.join(result)+')'

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
        if bond.neighbour==0:
            dgr=config[bond.epoint.pid]
            assert dgr.nspin==2
            ndim=dgr.norbital*dgr.nspin
            result=zeros((ndim,ndim,ndim,ndim),dtype=dtype)
            if self.atom is None or self.atom==dgr.atom:
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
        else:
            return 0

    def operators(self,bond,config,table=None,half=True,dtype=float64,**karg):
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
        half : logical, optional
            When True, only one half of the operators are returned, which means
                * The Hermitian conjugate of non-Hermitian operators is not included;
                * The coefficient of the self-Hermitian operators is divided by a factor 2.
        dtype: dtype,optional
            The data type of the coefficient of the returned operators.
 
        Returns
        -------
        Operators
            All the Hubbard operators with non-zero coefficients.
        '''
        result=Operators()
        dgr=config[bond.epoint.pid]
        mesh=self.mesh(bond,config,dtype=dtype)
        indices=argwhere(abs(mesh)>RZERO)
        for (i,j,k,l) in indices:
            index1=Index(bond.epoint.pid,dgr.state_index(i))
            index2=Index(bond.epoint.pid,dgr.state_index(j))
            index3=Index(bond.epoint.pid,dgr.state_index(k))
            index4=Index(bond.epoint.pid,dgr.state_index(l))
            if table is None:
                result+=FHubbard(
                    value=      mesh[i,j,k,l],
                    indices=    (index1.replace(nambu=CREATION),index2.replace(nambu=CREATION),index3,index4),
                    seqs=       None,
                    rcoord=     bond.epoint.rcoord,
                    icoord=     bond.epoint.icoord
                    )
            else:
                masks=next(iter(table)).masks
                temp1=index1.mask(*masks)
                temp2=index2.mask(*masks)
                temp3=index3.mask(*masks)
                temp4=index4.mask(*masks)
                if temp1 in table and temp2 in table and temp3 in table and temp4 in table:
                    result+=FHubbard(
                        value=      mesh[i,j,k,l],
                        indices=    (index1.replace(nambu=CREATION),index2.replace(nambu=CREATION),index3,index4),
                        seqs=       (table[temp1],table[temp2],table[temp3],table[temp4]),
                        rcoord=     bond.epoint.rcoord,
                        icoord=     bond.epoint.icoord
                        )
        if not half: result+=result.dagger
        return result
