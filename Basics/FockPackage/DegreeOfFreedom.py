'''
----------------------------------------
Fermionic and bosonic degrees of freedom
----------------------------------------

Fermionic and bosonic degree of freedom package, including:
    * constants: ANNIHILATION, CREATION, DEFAULT_FOCK_PRIORITY, DEFAULT_FERMIONIC_PRIORITY, DEFAULT_BOSONIC_PRIORITY
    * classes: FID, Fock, FockPack
    * functions: sigma0, sigmax, sigmay, sigmaz, sigmap,sigmam
'''

__all__=[   'ANNIHILATION','CREATION','DEFAULT_FOCK_PRIORITY','DEFAULT_FERMIONIC_PRIORITY','DEFAULT_BOSONIC_PRIORITY',
            'FID','Fock','FockPack',
            'sigma0','sigmax','sigmay','sigmaz','sigmap','sigmam'
            ]

from numpy.linalg import norm
from ..Utilities import RZERO,decimaltostr
from ..Geometry import PID
from ..DegreeOfFreedom import *
from copy import copy
from collections import namedtuple

ANNIHILATION,CREATION=0,1
DEFAULT_FOCK_PRIORITY=('scope','nambu','site','orbital','spin')
DEFAULT_FERMIONIC_PRIORITY=('scope','nambu','site','orbital','spin')
DEFAULT_BOSONIC_PRIORITY=('scope','nambu','site','orbital','spin')

class FID(namedtuple('FID',['orbital','spin','nambu'])):
    '''
    Fock space ID of fermionic/bosonic single particle Hilbert space.

    Attributes
    ----------
    orbital : int
        The orbital index, start with 0, default value 0. 
    spin : int
        The spin index, start with 0, default value 0.
    nambu : int
        '0' for ANNIHILATION and '1' for CREATION, default value ANNIHILATION.
    '''

    @property
    def dagger(self):
        '''
        The dagger of the fid.
        '''
        return self._replace(nambu=1-self.nambu)

FID.__new__.__defaults__=(0,0,ANNIHILATION)

class Fock(Internal):
    '''
    This class defines the internal fermionic/bosonic degrees of freedom in a single point.

    Attributes
    ----------
    atom : int
        The atom species on this point.
    norbital : int
        Number of orbitals.
    nspin : int
        Number of spins.
    nnambu : 1/2
        An integer to indicate whether or not using the Nambu space.
        1 means no and 2 means yes.
    '''

    def __init__(self,atom=0,norbital=1,nspin=2,nnambu=1):
        '''
        Constructor.

        Parameters
        ----------
        atom : int, optional
            The atom species.
        norbital : int, optional
            Number of orbitals.
        nspin : int, optional
            Number of spins.
        nnambu : 1/2, optional.
            A number to indicate whether or not the Nambu space is used.
            1 means no and 2 means yes.
        '''
        self.atom=atom
        self.norbital=norbital
        self.nspin=nspin
        self.nnambu=nnambu

    def __repr__(self):
        '''
        Convert an instance to string.
        '''
        return 'Fock(atom=%s, norbital=%s, nspin=%s, nnambu=%s)'%(self.atom,self.norbital,self.nspin,self.nnambu)

    def __eq__(self,other):
        '''
        Overloaded operator(==).
        '''
        return self.atom==other.atom and self.norbital==other.norbital and self.nspin==other.nspin and self.nnambu==other.nnambu

    def indices(self,pid,mask=()):
        '''
        Return a list of all the masked indices within this internal degrees of freedom combined with an extra spatial part.

        Parameters
        ----------
        pid : PID
            The extra spatial part of the indices.
        mask : list of str, optional
            The attributes that will be masked to None.

        Returns
        -------
        list of Index
            The indices.
        '''
        result=[]
        pid=pid._replace(**{key:None for key in set(mask)&set(PID._fields)})
        for nambu in (None,) if 'nambu' in mask else xrange(self.nnambu):
            for spin in (None,) if 'spin' in mask else xrange(self.nspin):
                for orbital in (None,) if 'orbital' in mask else xrange(self.norbital):
                    result.append(Index(pid=pid,iid=FID(orbital=orbital,spin=spin,nambu=nambu)))
        return result

class FockPack(IndexPack):
    '''
    This class is a part of a systematic description of a general fermionic/bosonic quadratic term.

    Attributes
    ----------
    atoms : 2-tuple of int, optional
        The atom indices of a quadratic term.
    orbitals : 2-tuple of int, optional
        The orbital indices of a quadratic term.
    spins : 2-tuple of int, optional
        The spin indices of a quadratic term.
    nambus : 2-tuple of int, optional
        The nambu indices of a quadratic term.
    '''

    def __init__(self,value=1.0,atoms=None,orbitals=None,spins=None,nambus=None):
        '''
        Constructor.

        Parameters
        ----------
        value : float or complex, optional
            The overall coefficient of the Fock pack
        atoms : 2-tuple of int, optional
            The atom indices.
        orbitals : 2-tuple of int, optional
            The orbital indices.
        spins : 2-tuple of int, optional
            The spin indices.
        nambus : 2-tuple of int, optional
            The nambu indices.
        '''
        super(FockPack,self).__init__(value)
        if atoms is not None:
            assert len(atoms)==2
            self.atoms=tuple(atoms)
        if orbitals is not None:
            assert len(orbitals)==2
            self.orbitals=tuple(orbitals)
        if spins is not None:
            assert len(spins)==2
            self.spins=tuple(spins)
        if nambus is not None:
            assert len(nambus)==2
            self.nambus=tuple(nambus)

    def tostr(self,mask=(),form='repr'):
        '''
        Convert an instance to string.

        Parameters
        ----------
        mask : tuple with elements from ('atoms','orbitals','spins','nambus'), optional
            The mask for the attributes of the Fock pack.
        form : 'repr' or 'str', optional
            The form of the string representation.

        Returns
        -------
        str
            The string representation of the Fock pack.
        '''
        assert form in ('repr','str')
        if form=='repr':
            condition=isinstance(self.value,complex) and abs(self.value.real)>5*10**-6 and abs(self.value.imag)>5*10**-6
            temp=['(%s)'%decimaltostr(self.value)] if condition else [decimaltostr(self.value)]
            if hasattr(self,'atoms') and 'atoms' not in mask: temp.append('sl%s%s'%self.atoms)
            if hasattr(self,'orbitals') and 'orbitals' not in mask: temp.append('ob%s%s'%self.orbitals)
            if hasattr(self,'spins') and 'spins' not in mask: temp.append('sp%s%s'%self.spins)
            if hasattr(self,'nambus') and 'nambus' not in mask: temp.append('ph%s%s'%self.nambus)
            return '*'.join(temp)
        else:
            temp=['value=%s'%self.value]
            if hasattr(self,'atoms') and 'atoms' not in mask: temp.append('atoms='+str(self.atoms))
            if hasattr(self,'orbitals') and 'orbitals' not in mask: temp.append('orbitals='+str(self.orbitals))
            if hasattr(self,'spins') and 'spins' not in mask: temp.append('spins='+str(self.spins))
            if hasattr(self,'nambus') and 'nambus' not in mask: temp.append('nambus='+str(self.nambus))
            return ''.join(['FockPack(',', '.join(temp),')'])

    def __repr__(self):
        '''
        Convert an instance to string.
        '''
        return self.tostr(form='repr')

    def __str__(self):
        '''
        Convert an instance to string.
        '''
        return self.tostr(form='str')

    def __mul__(self,other):
        '''
        Overloaded operator(*), which supports the multiplication of an instance of FockPack with an instance of FockPack/IndexPacks or a scalar.
        '''
        def MUL(self,other):
            if isinstance(other,FockPack):
                delta=lambda i,j: 1.0 if i==j else 0.0
                result=FockPack(self.value*other.value)
                for attr in ('atoms','orbitals','spins','nambus'):
                    if hasattr(self,attr) and hasattr(other,attr):
                        setattr(result,attr,(getattr(self,attr)[0],getattr(other,attr)[1]))
                        result.value*=delta(getattr(self,attr)[1],getattr(other,attr)[0])
                    elif hasattr(self,attr):
                        setattr(result,attr,getattr(self,attr))
                    elif hasattr(other,attr):
                        setattr(result,attr,getattr(other,attr))
            else:
                result=copy(self)
                result.value=self.value*other
            return result
        if isinstance(other,IndexPacks):
            result=IndexPacks()
            for fpack in other:
                temp=MUL(self,fpack)
                if norm(temp.value)>RZERO: result.append(temp)
        else:
            result=MUL(self,other)
        return result

    def __eq__(self,other):
        '''
        Overloaded operator(==).
        '''
        return (    self.value==other.value and 
                    getattr(self,'atoms',None)==getattr(other,'atoms',None) and 
                    getattr(self,'orbitals',None)==getattr(other,'orbitals',None) and 
                    getattr(self,'spins',None)==getattr(other,'spins',None) and 
                    getattr(self,'nambus',None)==getattr(other,'nambus',None)
                    )

    def expand(self,bond,sdgr,edgr):
        '''
        Expand the quadratics of the Fock pack on a bond.

        Parameters
        ----------
        bond : Bond
            The bond on which the expansion is performed.
        sdgr,edgr : Fock
            The internal degrees of freedom of the start point and end point of the bond.

        Returns
        -------
        generator of tuples in the form (value,index1,index2)
            * value : float or complex
                The coefficient of the quadratic.
            * index1,index2 : Index
                The indices of the quadratic.
        '''
        if not hasattr(self,'atoms') or (edgr.atom,sdgr.atom)==self.atoms:
            enambu,snambu=self.nambus if hasattr(self,'nambus') else (CREATION,ANNIHILATION)
            if hasattr(self,'spins'):
                if hasattr(self,'orbitals'):
                    index1=Index(bond.epoint.pid,FID(self.orbitals[0],self.spins[0],enambu))
                    index2=Index(bond.spoint.pid,FID(self.orbitals[1],self.spins[1],snambu))
                    yield self.value,index1,index2
                else:
                    assert edgr.norbital==sdgr.norbital
                    for k in xrange(edgr.norbital):
                        index1=Index(bond.epoint.pid,FID(k,self.spins[0],enambu))
                        index2=Index(bond.spoint.pid,FID(k,self.spins[1],snambu))
                        yield self.value,index1,index2
            else:
                assert edgr.nspin==sdgr.nspin
                if hasattr(self,'orbitals'):
                    for k in xrange(edgr.nspin):
                        index1=Index(bond.epoint.pid,FID(self.orbitals[0],k,enambu))
                        index2=Index(bond.spoint.pid,FID(self.orbitals[1],k,snambu))
                        yield self.value,index1,index2
                else:
                    assert edgr.norbital==sdgr.norbital
                    for k in xrange(edgr.nspin):
                        for j in xrange(edgr.norbital):
                            index1=Index(bond.epoint.pid,FID(j,k,enambu))
                            index2=Index(bond.spoint.pid,FID(j,k,snambu))
                            yield self.value,index1,index2
        else:
            return
            yield

def sigma0(mode):
    '''
    The 2-dimensional identity matrix, which can act on the space of spins('sp'), orbitals('ob'), sublattices('sl') or particle-holes('ph').
    '''
    result=IndexPacks()
    if mode.lower()=='sp':
        result.append(FockPack(1.0,spins=(0,0)))
        result.append(FockPack(1.0,spins=(1,1)))
    elif mode.lower()=='ob':
        result.append(FockPack(1.0,orbitals=(0,0)))
        result.append(FockPack(1.0,orbitals=(1,1)))
    elif mode.lower()=='sl':
        result.append(FockPack(1.0,atoms=(0,0)))
        result.append(FockPack(1.0,atoms=(1,1)))
    elif mode.lower()=='ph':
        result.append(FockPack(1.0,nambus=(ANNIHILATION,CREATION)))
        result.append(FockPack(1.0,nambus=(CREATION,ANNIHILATION)))
    else:
        raise ValueError("sigma0 error: mode '%s' not supported, which must be 'sp', 'ob', 'sl' or 'ph'."%mode)
    return result

def sigmax(mode):
    '''
    The Pauli matrix sigmax, which can act on the space of spins('sp'), orbitals('ob'), sublattices('sl') or particle-holes('ph').
    '''
    result=IndexPacks()
    if mode.lower()=='sp':
        result.append(FockPack(1.0,spins=(0,1)))
        result.append(FockPack(1.0,spins=(1,0)))
    elif mode.lower()=='ob':
        result.append(FockPack(1.0,orbitals=(0,1)))
        result.append(FockPack(1.0,orbitals=(1,0)))
    elif mode.lower()=='sl':
        result.append(FockPack(1.0,atoms=(0,1)))
        result.append(FockPack(1.0,atoms=(1,0)))
    elif mode.lower()=='ph':
        result.append(FockPack(1.0,nambus=(ANNIHILATION,ANNIHILATION)))
        result.append(FockPack(1.0,nambus=(CREATION,CREATION)))
    else:
        raise ValueError("sigmax error: mode '%s' not supported, which must be 'sp', 'ob', 'sl' or 'ph'."%mode)
    return result

def sigmay(mode):
    '''
    The Pauli matrix sigmay, which can act on the space of spins('sp'), orbitals('ob'), sublattices('sl') or particle-holes('ph').
    '''
    result=IndexPacks()
    if mode.lower()=='sp':
        result.append(FockPack(1.0j,spins=(0,1)))
        result.append(FockPack(-1.0j,spins=(1,0)))
    elif mode.lower()=='ob':
        result.append(FockPack(1.0j,orbitals=(0,1)))
        result.append(FockPack(-1.0j,orbitals=(1,0)))
    elif mode.lower()=='sl':
        result.append(FockPack(1.0j,atoms=(0,1)))
        result.append(FockPack(-1.0j,atoms=(1,0)))
    elif mode.lower()=='ph':
        result.append(FockPack(1.0j,nambus=(ANNIHILATION,ANNIHILATION)))
        result.append(FockPack(-1.0j,nambus=(CREATION,CREATION)))
    else:
        raise ValueError("sigmay error: mode '%s' not supported, which must be 'sp', 'ob', 'sl' or 'ph'."%mode)
    return result

def sigmaz(mode):
    '''
    The Pauli matrix sigmaz, which can act on the space of spins('sp'), orbitals('ob'), sublattices('sl') or particle-holes('ph').
    '''
    result=IndexPacks()
    if mode.lower()=='sp':
        result.append(FockPack(-1.0,spins=(0,0)))
        result.append(FockPack(1.0,spins=(1,1)))
    elif mode.lower()=='ob':
        result.append(FockPack(-1.0,orbitals=(0,0)))
        result.append(FockPack(1.0,orbitals=(1,1)))
    elif mode.lower()=='sl':
        result.append(FockPack(-1.0,atoms=(0,0)))
        result.append(FockPack(1.0,atoms=(1,1)))
    elif mode.lower()=='ph':
        result.append(FockPack(-1.0,nambus=(ANNIHILATION,CREATION)))
        result.append(FockPack(1.0,nambus=(CREATION,ANNIHILATION)))
    else:
        raise ValueError("sigmaz error: mode '%s' not supported, which must be 'sp', 'ob', 'sl' or 'ph'."%mode)
    return result

def sigmap(mode):
    '''
    The Pauli matrix sigma plus, which can act on the space of spins('sp'), orbitals('ob'), sublattices('sl') or particle-holes('ph').
    '''
    result=IndexPacks()
    if mode.lower()=='sp':
        result.append(FockPack(1.0,spins=(1,0)))
    elif mode.lower()=='ob':
        result.append(FockPack(1.0,orbitals=(1,0)))
    elif mode.lower()=='sl':
        result.append(FockPack(1.0,atoms=(1,0)))
    elif mode.lower()=='ph':
        result.append(FockPack(1.0,nambus=(CREATION,CREATION)))
    else:
        raise ValueError("sigmap error: mode '%s' not supported, which must be 'sp', 'ob', 'sl' or 'ph'."%mode)
    return result

def sigmam(mode):
    '''
    The Pauli matrix sigma minus, which can act on the space of spins('sp'), orbitals('ob'), sublattices('sl') or particle-holes('ph').
    '''
    result=IndexPacks()
    if mode.lower()=='sp':
        result.append(FockPack(1.0,spins=(0,1)))
    elif mode.lower()=='ob':
        result.append(FockPack(1.0,orbitals=(0,1)))
    elif mode.lower()=='sl':
        result.append(FockPack(1.0,atoms=(0,1)))
    elif mode.lower()=='ph':
        result.append(FockPack(1.0,nambus=(ANNIHILATION,ANNIHILATION)))
    else:
        raise ValueError("sigmam error: mode '%s' not supported, which must be 'sp', 'ob', 'sl' or 'ph'."%mode)
    return result
