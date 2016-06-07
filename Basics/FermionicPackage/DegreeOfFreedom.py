'''
Fermionic degree of freedom package, including:
1) constants: ANNIHILATION, CREATION, DEFAULT_FERMIONIC_PRIORITY
2) classes: FID, Fermi, IndexPackage, IndexPackageList
3) functions: sigmax, sigmay, sigmaz
'''

__all__=['ANNIHILATION','CREATION','DEFAULT_FERMIONIC_PRIORITY','FID','Fermi','IndexPackage','IndexPackageList','sigmax','sigmay','sigmaz']

from numpy import *
from ..DegreeOfFreedom import *
from copy import copy
from collections import namedtuple

ANNIHILATION,CREATION=0,1
DEFAULT_FERMIONIC_PRIORITY=['scope','nambu','spin','site','orbital']

class FID(namedtuple('FID',['orbital','spin','nambu'])):
    '''
    Internal fermionic ID.
    Attributes:
        orbital: integer
            The orbital index, start with 0, default value 0. 
        spin: integer
            The spin index, start with 0, default value 0.
        nambu: integer
            '0' for ANNIHILATION and '1' for CREATION, default value ANNIHILATION.
    '''

    @property
    def dagger(self):
        '''
        The dagger of the fermionic ID, i.e. sets nambu from CREATION to ANNIHILATION or vice versa while keeps others unchanged.
        '''
        return self._replace(nambu=1-self.nambu)

FID.__new__.__defaults__=(0,0,ANNIHILATION)

class Fermi(Internal):
    '''
    This class defines the internal fermionic degrees of freedom in a single point.
    Attributes:
        atom: integer, default value 0
            The atom species on this point.
        norbital: integer, default value 1
            Number of orbitals.
        nspin: integer, default value 2
            Number of spins.
        nnambu: integer, default value 1.
            An integer to indicate whether or not using the Nambu space.
            1 means no and 2 means yes.
    '''

    def __init__(self,atom=0,norbital=1,nspin=2,nnambu=1):
        '''
        Constructor.
            atom: integer, optional
                The atom species.
            norbital: integer, optional
                Number of orbitals.
            nspin: integer, optional
                Number of spins.
            nnambu: integer, optional.
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
        return 'Fermi(Atom=%s, norbital=%s, nspin=%s, nnambu=%s)'%(self.atom,self.norbital,self.nspin,self.nnambu)

    def __eq__(self,other):
        '''
        Overloaded operator(==).
        '''
        return self.atom==other.atom and self.norbital==other.norbital and self.nspin==other.nspin and self.nnambu==other.nnambu

    def table(self,pid,nambu=False,key=None):
        '''
        This method returns a Table instance that contains all the allowed indices constructed from an input pid and the internal degrees of freedom.
        Parameters:
            pid: PID
                The spatial part of the indices.
            nambu: logical, optional
                A flag to tag whether or not the nambu space is used.
            key: function
                The key function used to sort the indices.
        Returns: Table
            The index-sequence table.
        '''
        result=[]
        if nambu:
            for buff in xrange(self.nnambu):
                for spin in xrange(self.nspin):
                    for orbital in xrange(self.norbital):
                        result.append(Index(pid=pid,iid=FID(orbital=orbital,spin=spin,nambu=buff)))
        else:
            for spin in xrange(self.nspin):
                for orbital in xrange(self.norbital):
                    result.append(Index(pid=pid,iid=FID(orbital=orbital,spin=spin,nambu=ANNIHILATION)))
        if key is None:
            return Table(result)
        else:
            return Table(sorted(result,key=key))

    def seq_state(self,fid):
        '''
        This methods is the oversimplified version of returning the sequence of a input state with orbital, spin and nambu index assigned.
        Note: the priority to generate the sequence cannot be modified by the users and is always "NSO".
        '''
        if fid.nambu in (0,1):
            return fid.orbital+fid.spin*self.norbital+fid.nambu*self.norbital*self.nspin
        else:
            raise ValueError("Fermi seq_state error: the nambu index must be 0 or 1.")

    def state_index(self,seq_state):
        '''
        This methods returns an instance of FID that contains the orbital, spin and nambu index of a state whose sequence equals the input seq_state.
        Parameters:
            seq_state: integer
                The sequence of the state.
        Returns: FID
            The corresponding FID.
        Note: This method should be used in pairs with the method seq_state to ensure the correct sequence-index correspondence.
        '''
        spin=seq_state%(self.norbital*self.nspin)/self.norbital
        orbital=seq_state%(self.norbital*self.nspin)%self.norbital
        nambu=seq_state/(self.norbital*self.nspin)
        return FID(spin=spin,orbital=orbital,nambu=nambu)

class IndexPackage:
    '''
    This class assumes part of a systematic description of a general fermionic quadratic term.
    Attributes:
        value: float or complex
            The overall coefficient of the index pack.
        atoms: tuple of integers with len==2
            The atom indices for the quadratic term.
        orbitals: tuple of integers with len==2
            The orbital indices for the quadratic term.
        spins: tuple of integers with len==2
            The spin indices for the quadratic term.
    '''
    
    def __init__(self,value,atom1=None,atom2=None,orbital1=None,orbital2=None,spin1=None,spin2=None,atoms=None,orbitals=None,spins=None):
        '''
        Constructor. 
        It can be used in two different ways:
        1) IndexPackage(value,atom1=...,atom2=...,orbital1=...,orbital2=...,spin1=...,spin2=...)
        2) IndexPackage(value,atoms=...,orbitals=...,spins=...)
        Parameters:
            value: float or complex
                The overall coefficient of the index pack
            atom1,atom2: integer,optional
                The atom indices.
            orbital1,orbital2: integer,optional
                The orbital indices.
            spin1,spin2: integer, optional
                The spin indices.
            atoms: 1D array-like of integers with len==1,2,optional
                The atom indices.
            orbitals: 1D array-like of integers with len==1,2,optional
                The orbital indices.
            spins: 1D array-like of integers with len==1,2,optional
                The spin indices.
        '''
        self.value=value
        if atom1 is not None and atom2 is not None: self.atoms=(atom1,atom2)
        if orbital1 is not None and orbital2 is not None: self.orbitals=(orbital1,orbital2)
        if spin1 is not None and spin2 is not None: self.spins=(spin1,spin2)
        if atoms is not None:
            if len(atoms)==2: self.atoms=tuple(atoms)
            elif len(atoms)==1: self.atoms=(atoms[0],atoms[0])
        if orbitals is not None:
            if len(orbitals)==2: self.orbitals=tuple(orbitals)
            elif len(orbitals)==1: self.orbitals=([orbitals[0],orbitals[0]])
        if spins is not None:
            if len(spins)==2: self.spins=tuple(spins)
            elif len(spins)==1: self.spins=([spins[0],spins[0]])

    def __repr__(self):
        '''
        Convert an instance to string.
        '''
        temp=[]
        temp.append('value=%s'%self.value)
        if hasattr(self,'atoms'):
            temp.append('atoms='+str(self.atoms))
        if hasattr(self,'orbitals'):
            temp.append('orbitals='+str(self.orbitals))
        if hasattr(self,'spins'):
            temp.append('spins='+str(self.spins))
        return ''.join(['IndexPackage(',', '.join(temp),')'])

    def __add__(self,other):
        '''
        Overloaded operator(+), which supports the addition of an IndexPackage instance with an IndexPackage/IndexPackageList instance.
        '''
        result=IndexPackageList()
        result.append(self)
        if isinstance(other,IndexPackage):
            result.append(other)
        elif isinstance(other,IndexPackageList):
            result.extend(other)
        else:
            raise ValueError("IndexPackage '+' error: the 'other' parameter must be of class IndexPackage or IndexPackageList.")
        return result

    def _mul(self,other):
        '''
        Private methods used for operator(*) overloading.
        '''
        if isinstance(other,IndexPackage):
            result=IndexPackage(self.value*other.value)
            if hasattr(self,'atoms'): result.atoms=self.atoms
            if hasattr(self,'orbitals'): result.orbitals=self.orbitals
            if hasattr(self,'spins'): result.spins=self.spins
            if hasattr(other,'atoms'):
                if not hasattr(result,'atoms'):
                    result.atoms=other.atoms
                else:
                    raise ValueError("IndexPackage '*' error: 'self' and 'other' cannot simultaneously have the 'atoms' attribute.")
            if hasattr(other,'orbitals'):
                if not hasattr(result,'orbitals'):
                    result.orbitals=other.orbitals
                else:
                    raise ValueError("IndexPackage '*' error: 'self' and 'other' cannot simultaneously have the 'orbitals' attribute.")
            if hasattr(other,'spins'):
                if not hasattr(result,'spins'):
                    result.spins=other.spins
                else:
                    raise ValueError("IndexPackage '*' error: 'self' and 'other' cannot simultaneously have the 'spins' attribute.")
        else:
            result=copy(self)
            result.value=self.value*other
        return IndexPackageList(result)
    
    def __mul__(self,other):
        '''
        Overloaded operator(*), which supports the multiplication of an IndexPackage instance with an IndexPackage/IndexPackageList instance or a scalar.
        '''
        result=IndexPackageList()
        if isinstance(other,IndexPackage) or type(other)==int or type(other)==long or type(other)==float or type(other)==complex:
            result.extend(self._mul(other))
        elif isinstance(other,IndexPackageList):
            for buff in other:
                result.extend(self._mul(buff))
        else:
            raise ValueError("IndexPackage '*' error: the 'other' parameter must be of class IndexPackage or IndexPackageList, or a number.")
        return result

class IndexPackageList(list):
    '''
    This class packs several IndexPackage as a whole for convenience.
    '''
    def __init__(self,*arg):
        for buff in arg:
            if isinstance(buff,IndexPackage):
                self.append(buff)
            else:
                raise ValueError("IndexPackageList init error: the input parameters must be of class IndexPackage.")

    def __str__(self):
        '''
        Convert an instance to string.
        '''
        return 'IndexPackageList('+', '.join([str(obj) for obj in self])
                
    def __add__(self,other):
        '''
        Overloaded operator(+), which supports the addition of an IndexPackageList instance with an IndexPackage/IndexPackageList instance.
        '''
        result=IndexPackageList(*self)
        if isinstance(other,IndexPackage):
            result.append(other)
        elif isinstance(other,IndexPackageList):
            result.extend(other)
        else:
            raise ValueError("IndexPackageList '+' error: the 'other' parameter must be of class IndexPackage or IndexPackageList.")
        return result
    
    def __mul__(self,other):
        '''
        Overloaded operator(*), which supports the multiplication of an IndexPackageList instance with an IndexPackage/IndexPackageList instance or a scalar.
        '''
        result=IndexPackageList()
        if isinstance(other,IndexPackage) or isinstance(other,IndexPackageList) or type(other)==int or type(other)==long or type(other)==float or type(other)==complex:
            for buff in self:
                result.extend(buff*other)
        else:
            raise ValueError("IndexPackageList '*' error: the 'other' parameter must be of class IndexPackage or IndexPackageList, or a number.")
        return result

def sigmax(mode):
    '''
    The Pauli matrix SigmaX, which can act on the space of spins('sp'), orbitals('ob') or sublattices('sl').
    '''
    result=IndexPackageList()
    if mode.lower()=='sp':
        result.append(IndexPackage(1.0,spin1=0,spin2=1))
        result.append(IndexPackage(1.0,spin1=1,spin2=0))
    elif mode.lower()=='ob':
        result.append(IndexPackage(1.0,orbital1=0,orbital2=1))
        result.append(IndexPackage(1.0,orbital1=1,orbital2=0))
    elif mode.lower()=='sl':
        result.append(IndexPackage(1.0,atom1=0,atom2=1))
        result.append(IndexPackage(1.0,atom1=1,atom2=0))
    else:
        raise ValueError("SigmaX error: mode '%s' not supported, which must be 'sp', 'ob', or 'sl'."%mode)
    return result

def sigmay(mode):
    '''
    The Pauli matrix SigmaY, which can act on the space of spins('sp'), orbitals('ob') or sublattices('sl').
    '''
    result=IndexPackageList()
    if mode.lower()=='sp':
        result.append(IndexPackage(-1.0j,spin1=0,spin2=1))
        result.append(IndexPackage(1.0j,spin1=1,spin2=0))
    elif mode.lower()=='ob':
        result.append(IndexPackage(-1.0j,orbital1=0,orbital2=1))
        result.append(IndexPackage(1.0j,orbital1=1,orbital2=0))
    elif mode.lower()=='sl':
        result.append(IndexPackage(-1.0j,atom1=0,atom2=1))
        result.append(IndexPackage(1.0j,atom1=1,atom2=0))
    else:
        raise ValueError("SigmaY error: mode '%s' not supported, which must be 'sp', 'ob', or 'sl'."%mode)
    return result

def sigmaz(mode):
    '''
    The Pauli matrix SigmaZ, which can act on the space of spins('sp'), orbitals('ob') or sublattices('sl').
    '''
    result=IndexPackageList()
    if mode.lower()=='sp':
        result.append(IndexPackage(1.0,spin1=0,spin2=0))
        result.append(IndexPackage(-1.0,spin1=1,spin2=1))
    elif mode.lower()=='ob':
        result.append(IndexPackage(1.0,orbital1=0,orbital2=0))
        result.append(IndexPackage(-1.0,orbital1=1,orbital2=1))
    elif mode.lower()=='sl':
        result.append(IndexPackage(1.0,atom1=0,atom2=0))
        result.append(IndexPackage(-1.0,atom1=1,atom2=1))
    else:
        raise ValueError("SigmaZ error: mode '%s' not supported, which must be 'sp', 'ob', or 'sl'."%mode)
    return result
