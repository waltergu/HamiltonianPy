'''
---------------
Fermionic terms
---------------

Fermionic terms, including:
    * classes: Quadratic, Hubbard, Coulomb
    * functions: Hopping, Onsite, Pairing
'''

__all__=['Quadratic','Hopping','Onsite','Pairing','Hubbard','Coulomb']

from ..Utilities import RZERO,decimaltostr
from ..Term import *
from ..DegreeOfFreedom import *
from ..Operator import *
from ..Geometry import Bond
from DegreeOfFreedom import *
from Operator import * 
import numpy as np

class Quadratic(Term):
    '''
    This class provides a complete and unified description for fermionic/bosonic quadratic terms, e.g. hopping terms, onsite terms and pairing terms.

    Attributes
    ----------
    statistics : 'f', 'b'
        The statistics of the particles involved in the quadratic term, 'f' for fermionic and 'b' for bosonic.
    mode : 'hp','st','pr'
        The type of the term, 'hp' for hopping, 'st' for onsite and 'pr' for pairing.
    neighbour : int
        The order of neighbour of this quadratic term.
    indexpacks : IndexPacks or callable which returns IndexPacks
        The indexpacks of the quadratic term.
        When it is callable, it returns bond dependent indexpacks as needed.
    amplitude : callable which returns float or complex
        This function returns bond dependent coefficient as needed.

    Notes
    -----
    The final coefficient comes from three parts, the value of itself, the value of the indexpack, and the value amplitude returns.
    '''

    def __init__(self,id,statistics,mode,value=1.0,neighbour=0,atoms=None,orbitals=None,spins=None,nambus=None,indexpacks=None,amplitude=None,modulate=None):
        '''
        Constructor.

        Parameters
        ----------
        id : str
            The specific id of the term.
        statistics : 'f', 'b'
            The statistics of the particles involved in the quadratic term, 'f' for fermionic and 'b' for bosonic.
        mode : 'hp','st','pr'
            The type of the term, 'hp' for hopping, 'st' for onsite and 'pr' for pairing.
        value : float or complex
            The overall coefficient of the term.
        neighbour : int, optional
            The order of neighbour of the term.
        atoms,orbitals,spins,nambus : 2-tuple of int, optional
            The atom, orbital, spin and nambu indices complementary to the indexpacks specific by the parameter `indexpacks`.
        indexpacks : IndexPacks or callable, optional
            * IndexPacks:
                The indexpacks of the quadratic term.
            * callable in the form ``indexpacks(bond)``:
                It returns the bond-dependent indexpacks of the quadratic term.
        amplitude: callable in the form ``amplitude(bond)``, optional
            It returns the bond-dependent amplitude of the quadratic term.
        modulate: callable in the form ``modulate(*arg,**karg)``, optional
            This function defines the way to change the overall coefficient of the term dynamically.
        '''
        assert statistics in ('f','b') and mode in ('hp','st','pr')
        super(Quadratic,self).__init__(id=id,value=value,modulate=modulate)
        self.statistics=statistics
        self.mode=mode
        self.neighbour=neighbour
        fpack=FockPack(1.0,atoms=atoms,orbitals=orbitals,spins=spins,nambus=nambus)
        if indexpacks is None:
            self.indexpacks=IndexPacks(fpack)
        elif isinstance(indexpacks,IndexPacks):
            self.indexpacks=fpack*indexpacks
        elif callable(indexpacks):
            self.indexpacks=lambda bond: fpack*indexpacks(bond)
        else:
            raise ValueError('Quadratic init error: the input indexpacks should be an instance of IndexPacks or a function.')
        self.amplitude=amplitude

    def __repr__(self):
        '''
        Convert an instance to string.
        '''
        result=[]
        result.append('id=%s'%self.id)
        result.append('statistics=%s'%self.statistics)
        result.append('mode=%s'%self.mode)
        result.append('value=%s'%self.value)
        result.append('neighbour=%s'%self.neighbour)
        result.append('indexpacks=%s'%self.indexpacks)
        if self.amplitude is not None:
            result.append('amplitude=%s'%self.amplitude)
        if self.modulate is not None:
            result.append('modulate=%s'%self.modulate)
        return 'Quadratic(%s)'%(', '.join(result))

    def operators(self,bond,config,table=None,half=True,dtype=np.complex128,**karg):
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
                * The Hermitian conjugate of non-Hermitian operators are not included;
                * The coefficient of the self-Hermitian operators are divided by a factor 2.
        dtype : np.complex128, np.float64, optional
            The data type of the coefficient of the returned operators.

        Returns
        -------
        Operators
            The quadratic operators with non-zero coefficients.

        Notes
        -----
        No matter whether or not ``half`` is True, for the BdG case, only the electron part of the hopping terms and onsite terms are contained.
        '''
        def expansion(bond,config,half):
            result={}
            if self.neighbour==bond.neighbour:
                value=self.value*(1 if self.amplitude is None else self.amplitude(bond))
                for fpack in self.indexpacks(bond) if callable(self.indexpacks) else self.indexpacks:
                    if self.mode=='pr':
                        assert getattr(fpack,'nambus',None) in ((ANNIHILATION,ANNIHILATION),(CREATION,CREATION))
                    else:
                        assert not hasattr(fpack,'nambus')
                    for coeff,index1,index2 in fpack.expand(bond,config[bond.spoint.pid],config[bond.epoint.pid]):
                        dagger1,dagger2=index1.replace(nambu=1-index1.nambu),index2.replace(nambu=1-index2.nambu)
                        if half:
                            if self.mode=='st' and index1==dagger2:
                                result[(index1,index2)]=value*coeff/2+result.get((index1,index2),0.0)
                            elif self.mode in ('hp','pr') or (self.mode=='st' and (dagger2,dagger1) not in result):
                                result[(index1,index2)]=value*coeff+result.get((index1,index2),0.0)
                        else:
                            result[(index1,index2)]=value*coeff+result.get((index1,index2),0.0)
                            if self.mode in ('hp','pr'):
                                result[(dagger2,dagger1)]=np.conjugate(value*coeff)+result.get((dagger2,dagger1),0.0)
            return result
        CONSTRUCTOR=FQuadratic if self.statistics=='f' else BQuadratic
        def operators(expansion,bond,table):
            result=Operators()
            for (eindex,sindex),value in expansion.iteritems():
                if np.abs(value)>RZERO:
                    if table is None:
                        result+=CONSTRUCTOR(value=dtype(value),indices=(eindex,sindex),seqs=None,rcoord=bond.rcoord,icoord=bond.icoord)
                    else:
                        masks=next(iter(table)).masks
                        etemp=eindex.replace(nambu=1-eindex.nambu).mask(*masks)
                        stemp=sindex.mask(*masks)
                        if stemp in table and etemp in table:
                            result+=CONSTRUCTOR(value=dtype(value),indices=(eindex,sindex),seqs=(table[etemp],table[stemp]),rcoord=bond.rcoord,icoord=bond.icoord)
            return result
        result=operators(expansion(bond,config,half),bond,table)
        if self.mode=='pr':
            result+=operators(expansion(bond.reversed,config,half),bond.reversed,table)
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
                edgr,sdgr=config[bond.epoint.pid],config[bond.spoint.pid]
                for fpack in self.indexpacks(bond) if callable(self.indexpacks) else self.indexpacks:
                    if not hasattr(fpack,'atoms') or (edgr.atom,sdgr.atom)==fpack.atoms:
                        result.append('%s%s:%s*%s'%(self.statistics,self.mode,decimaltostr(value,Term.NDECIMAL),fpack.tostr(mask=('atoms',),form='repr')))
        return '\n'.join(result)

def Hopping(id,value,neighbour=1,atoms=None,orbitals=None,spins=None,indexpacks=None,amplitude=None,modulate=None,statistics='f'):
    '''
    A specified function to construct a hopping term.
    '''
    return Quadratic(id,statistics,'hp',value,neighbour,atoms,orbitals,spins,None,indexpacks,amplitude,modulate)

def Onsite(id,value,atoms=None,orbitals=None,spins=None,indexpacks=None,amplitude=None,modulate=None,statistics='f'):
    '''
    A specified function to construct an onsite term.
    '''
    return Quadratic(id,statistics,'st',value,0,atoms,orbitals,spins,None,indexpacks,amplitude,modulate)

def Pairing(id,value,neighbour=0,atoms=None,orbitals=None,spins=None,indexpacks=None,amplitude=None,modulate=None,statistics='f'):
    '''
    A specified function to construct an pairing term.
    '''
    return Quadratic(id,statistics,'pr',value,neighbour,atoms,orbitals,spins,(ANNIHILATION,ANNIHILATION),indexpacks,amplitude,modulate)

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
    statistics : 'f', 'b'
        The statistics of the particles involved in the Hubbard term, 'f' for fermionic and 'b' for bosonic.
    atom : int
        The atom index of the point where the Hubbard interactions are defined.
    '''

    def __init__(self,id,value=1.0,atom=None,modulate=None,statistics='f'):
        '''
        Constructor.
        '''
        assert statistics in ('f','b')
        super(Hubbard,self).__init__(id=id,value=value,modulate=modulate)
        self.statistics=statistics
        self.atom=atom

    def __repr__(self):
        '''
        Convert an instance to string.
        '''
        result=[]
        result.append('id=%s'%self.id)
        result.append('statistics=%s'%self.statistics)
        if self.atom is not None: result.append('atom=%s'%self.atom)
        result.append('value=%s'%self.value)
        return 'Hubbard(%s)'%(', '.join(result))

    def __len__(self):
        '''
        1 for single-orbital Hubbard term and 4 for multi-orbital Hubbard term.
        '''
        return 4 if hasattr(self.value,'__len__') else 1

    def operators(self,bond,config,table=None,half=True,order='normal',dtype=np.float64,**karg):
        '''
        This method returns all the Hubbard operators defined on the input bond with non-zero coefficients.

        Parameters
        ----------
        bond : Bond
            The bond on which the Hubbard terms are defined.
        config : IDFConfig
            The configuration of internal degrees of freedom.
        table : Table, optional
            The index-sequence table.
        half : logical, optional
            When True, only one half of the operators are returned, which means
                * The Hermitian conjugate of non-Hermitian operators are not included;
                * The coefficient of the self-Hermitian operators are divided by a factor 2.
        order : 'normal' or 'density'
            'normal' for normal ordered order and 'density' for density-density formed order.
        dtype : np.complex128, np.float64, optional
            The data type of the coefficient of the returned operators.
 
        Returns
        -------
        Operators
            All the Hubbard operators with non-zero coefficients.
        '''
        result=Operators()
        nv,pid,dgr=len(self),bond.epoint.pid,config[bond.epoint.pid]
        if bond.neighbour==0 and self.atom in (None,dgr.atom):
            assert nv in (1,4) and dgr.nspin==2 and order in ('normal','density')
            expansion=[]
            if nv>=1:
                for h in xrange(dgr.norbital):
                    value=self.value/2 if nv==1 else self.value[0]/2
                    index1=Index(pid,FID(h,1,CREATION))
                    index2=Index(pid,FID(h,0,CREATION))
                    index3=Index(pid,FID(h,0,ANNIHILATION))
                    index4=Index(pid,FID(h,1,ANNIHILATION))
                    expansion.append((value,index1,index2,index3,index4))
            if nv==4:
                for h in xrange(dgr.norbital):
                    for g in xrange(dgr.norbital):
                        if g!=h:
                            value=self.value[1]/2
                            index1=Index(pid,FID(g,1,CREATION))
                            index2=Index(pid,FID(h,0,CREATION))
                            index3=Index(pid,FID(h,0,ANNIHILATION))
                            index4=Index(pid,FID(g,1,ANNIHILATION))
                            expansion.append((value,index1,index2,index3,index4))
                for h in xrange(dgr.norbital):
                    for g in xrange(h):
                        for f in xrange(2):
                            value=(self.value[1]-self.value[2])/2
                            index1=Index(pid,FID(g,f,CREATION))
                            index2=Index(pid,FID(h,f,CREATION))
                            index3=Index(pid,FID(h,f,ANNIHILATION))
                            index4=Index(pid,FID(g,f,ANNIHILATION))
                            expansion.append((value,index1,index2,index3,index4))
                for h in xrange(dgr.norbital):
                    for g in xrange(h):
                        value=self.value[2]
                        index1=Index(pid,FID(g,1,CREATION))
                        index2=Index(pid,FID(h,0,CREATION))
                        index3=Index(pid,FID(g,0,ANNIHILATION))
                        index4=Index(pid,FID(h,1,ANNIHILATION))
                        expansion.append((value,index1,index2,index3,index4))
                for h in xrange(dgr.norbital):
                    for g in xrange(h):
                        value=self.value[3]
                        index1=Index(pid,FID(g,1,CREATION))
                        index2=Index(pid,FID(g,0,CREATION))
                        index3=Index(pid,FID(h,0,ANNIHILATION))
                        index4=Index(pid,FID(h,1,ANNIHILATION))
                        expansion.append((value,index1,index2,index3,index4))
            CONSTRUCTOR=FHubbard if self.statistics=='f' else BHubbard
            for value,index1,index2,index3,index4 in expansion:
                if np.abs(value)>RZERO:
                    if table is None:
                        result+=CONSTRUCTOR(
                                value=      dtype(value),
                                indices=    (index1,index2,index3,index4),
                                seqs=       None,
                                rcoord=     bond.epoint.rcoord,
                                icoord=     bond.epoint.icoord
                                )
                    else:
                        masks=next(iter(table)).masks
                        temp1=index1.replace(nambu=1-index1.nambu).mask(*masks)
                        temp2=index2.replace(nambu=1-index2.nambu).mask(*masks)
                        temp3=index3.mask(*masks)
                        temp4=index4.mask(*masks)
                        if temp1 in table and temp2 in table and temp3 in table and temp4 in table:
                            result+=CONSTRUCTOR(
                                value=      dtype(value),
                                indices=    (index1,index2,index3,index4),
                                seqs=       (table[temp1],table[temp2],table[temp3],table[temp4]),
                                rcoord=     bond.epoint.rcoord,
                                icoord=     bond.epoint.icoord
                                )
            if not half: result+=result.dagger
            if order=='density': result=Operators((opt.id,opt) for opt in [opt.reorder([0,3,1,2]) for opt in result.itervalues()])
        return result

    @property
    def unit(self):
        '''
        The unit term.
        '''
        if len(self)==1:
            return self.replace(value=1.0)
        else:
            raise TypeError('Hubbard unit error: not supported.')

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
        if bond.neighbour==0 and self.atom in (None,config[bond.epoint.pid].atom):
            return '%shb:%s'%(self.statistics,'_'.join(decimaltostr(value,Term.NDECIMAL) for value in (self.value if len(self)==4 else [self.value])))
        else:
            return ''

class Coulomb(Term):
    '''
    This class provides a complete description of Coulomb interactions.

    Attributes
    ----------
    statistics : 'f', 'b'
        The statistics of the particles involved in the Coulomb term, 'f' for fermionic and 'b' for bosonic.
    neighbour : int
        The order of neighbour of this Coulomb term.
    indexpacks : 2-tuple of IndexPacks or function which returns a 2-tuple of IndexPacks
        * 2-tuple of IndexPacks
            The indexpacks at the end point and start point of the Coulomb term.
        * function which returns a 2-tuple of IndexPacks
            This function returns bond dependent indexpacks at the end point and start point of the Coulomb term.
    amplitude : function which returns float or complex
        This function returns bond dependent coefficient as needed.
    '''

    def __init__(self,id,value=1.0,neighbour=0,indexpacks=None,amplitude=None,modulate=None,statistics='f'):
        '''
        Constructor.

        Parameters
        ----------
        id : str
            The specific id of the term.
        value : float or complex
            The overall coefficient of the term.
        neighbour : int, optional
            The order of neighbour of the term.
        indexpacks : 2-tuple of IndexPacks or callable which returns a 2-tuple of IndexPacks, optional
            * 2-tuple of IndexPacks
                The indexpacks at the end point and start point of the Coulomb term.
            * callable in the form ``indexpacks(bond)`` which returns a 2-tuple of IndexPacks
                This function returns bond dependent indexpacks at the end point and start point of the Coulomb term.
        amplitude: callable in the form ``amplitude(bond)``, optional
            It returns the bond-dependent amplitude of the Coulomb term.
        modulate: callable in the form ``modulate(*arg,**karg)``, optional
            This function defines the way to change the overall coefficient of the term dynamically.
        statistics : 'f', 'b', optional
            The statistics of the particles involved in the Coulomb term, 'f' for fermionic and 'b' for bosonic.
        '''
        assert statistics in ('f','b')
        super(Coulomb,self).__init__(id=id,value=value,modulate=modulate)
        self.statistics=statistics
        self.neighbour=neighbour
        if indexpacks is None:
            self.indexpacks=(IndexPacks(FockPack(value=1.0)),IndexPacks(FockPack(value=1.0)))
        elif callable(indexpacks):
            self.indexpacks=indexpacks
        else:
            assert len(indexpacks)==2
            self.indexpacks=tuple(indexpacks)
        self.amplitude=amplitude

    def __repr__(self):
        '''
        Convert an instance to string.
        '''
        result=[]
        result.append('id=%s'%self.id)
        result.append('value=%s'%self.value)
        result.append('statistics=%s'%self.statistics)
        result.append('neighbour=%s'%self.neighbour)
        result.append('indexpacks=%s'%(self.indexpacks if callable(self.indexpacks) else (self.indexpacks,)))
        if self.amplitude is not None:
            result.append('amplitude=%s'%self.amplitude)
        if self.modulate is not None:
            result.append('modulate=%s'%self.modulate)
        return 'Coulomb(%s)'%(', '.join(result))

    def operators(self,bond,config,table=None,half=True,dtype=np.complex128,**karg):
        '''
        This method returns all the Coulomb operators defined on the input bond with non-zero coefficients.

        Parameters
        ----------
        bond : Bond
            The bond on which the Coulomb term is defined.
        config : IDFConfig
            The configuration of internal degrees of freedom.
        table : Table, optional
            The index-sequence table.
        half : logical, optional
            When True, only one half of the operators are returned, which means
                * The Hermitian conjugate of non-Hermitian operators are not included;
                * The coefficient of the self-Hermitian operators are divided by a factor 2.
        dtype : np.complex128, np.float64, optional
            The data type of the coefficient of the returned operators.

        Returns
        -------
        Operators
            All the Coulomb operators with non-zero coefficients.
        '''
        result=Operators()
        if self.neighbour==bond.neighbour:
            expansion={}
            value=self.value*(1 if self.amplitude is None else self.amplitude(bond))
            eindexpacks,sindexpacks=self.indexpacks(bond) if callable(self.indexpacks) else self.indexpacks
            for epack in eindexpacks:
                for ecoeff,index1,index2 in epack.expand(Bond(0,bond.epoint,bond.epoint),config[bond.epoint.pid],config[bond.epoint.pid]):
                    dagger1,dagger2=index1.replace(nambu=1-index1.nambu),index2.replace(nambu=1-index2.nambu)
                    for spack in sindexpacks:
                        for scoeff,index3,index4 in spack.expand(Bond(0,bond.spoint,bond.spoint),config[bond.spoint.pid],config[bond.spoint.pid]):
                            dagger3,dagger4=index3.replace(nambu=1-index3.nambu),index4.replace(nambu=1-index4.nambu)
                            coeff,key=value*ecoeff*scoeff,(index1,index2,index3,index4)
                            if index1==dagger2 and index3==dagger4:
                                expansion[key]=coeff/(2.0 if half else 1.0)+expansion.get(key,0.0)
                            else:
                                expansion[key]=coeff+expansion.get(key,0.0)
                                if not half:
                                    key=(dagger4,dagger3,dagger2,dagger1) if self.neighbour==0 else (dagger2,dagger1,dagger4,dagger3)
                                    expansion[key]=np.conjugate(coeff)+expansion.get(key,0.0)
            CONSTRUCTOR=FCoulomb if self.statistics=='f' else BCoulomb
            for (index1,index2,index3,index4),value in expansion.iteritems():
                if np.abs(value)>RZERO:
                    if table is None:
                        result+=CONSTRUCTOR(
                                value=      dtype(value),
                                indices=    (index1,index2,index3,index4),
                                seqs=       None,
                                rcoord=     bond.rcoord,
                                icoord=     bond.icoord
                                )
                    else:
                        masks=next(iter(table)).masks
                        temp1=index1.replace(nambu=1-index1.nambu).mask(*masks)
                        temp2=index2.replace(nambu=1-index2.nambu).mask(*masks)
                        temp3=index3.mask(*masks)
                        temp4=index4.mask(*masks)
                        if temp1 in table and temp2 in table and temp3 in table and temp4 in table:
                            result+=CONSTRUCTOR(
                                value=      dtype(value),
                                indices=    (index1,index2,index3,index4),
                                seqs=       (table[temp1],table[temp2],table[temp3],table[temp4]),
                                rcoord=     bond.rcoord,
                                icoord=     bond.icoord
                                )
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
        result=''
        if self.neighbour==bond.neighbour:
            value=self.value*(1 if self.amplitude is None else self.amplitude(bond))
            if np.abs(value)>RZERO:
                edgr,sdgr=config[bond.epoint.pid],config[bond.spoint.pid]
                eindexpacks,sindexpacks=self.indexpacks(bond) if callable(self.indexpacks) else self.indexpacks
                epacks,spacks=[],[]
                for epack in eindexpacks:
                    if not hasattr(epack,'atoms') or (edgr.atom,edgr.atom)==epack.atoms:
                        epacks.append(epack.tostr(mask=('atoms',),form='repr'))
                epacks='%s%s%s'%('(' if len(epacks)>1 else '','+'.join(epacks),')' if len(epacks)>1 else '')
                for spack in sindexpacks:
                    if not hasattr(spack,'atoms') or (sdgr.atom,sdgr.atom)==spack.atoms:
                        spacks.append(spack.tostr(mask=('atoms',),form='repr'))
                spacks='%s%s%s'%('(' if len(spacks)>1 else '','+'.join(spacks),')' if len(spacks)>1 else '')
                result='%scl:%s*%s*%s'%(self.statistics,decimaltostr(value,Term.NDECIMAL),epacks,spacks)
        return result
