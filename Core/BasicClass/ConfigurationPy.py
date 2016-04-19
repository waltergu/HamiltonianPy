'''
Configurations of the internal degrees of freedom in a lattice, including:
1) constants: ANNIHILATION, CREATION
2) classes: Index, Internal, FID, Fermi,
'''

__all__=['ANNIHILATION','CREATION','Index','Internal','FID','Fermi']

from collections import namedtuple,OrderedDict

class Index(tuple):
    '''
    Index.
    '''
    def __new__(cls,pid,iid):
        '''
        Constructor.
        Parameters:
            pid: PID
                The point index, i.e. the spatial part in a lattice of the index
            iid: namedtuple
                The internal index, i.e. the internal part in a point of the index.
        '''
        self=super(Index,cls).__new__(cls,pid+iid)
        self.__dict__=pid._asdict()
        self.__dict__.update(iid._asdict())
        return self

    def __repr__(self):
        '''
        Convert an instance to string.
        '''
        return ''.join(['Index(','=%r, '.join(self.__dict__.keys()),'=%r)'])%self

    def replace(self,**karg):
        '''
        Return a new Index object with specified fields replaced with new values.
        '''
        result=tuple.__new__(Index,map(karg.pop,self.__dict__.keys(),self))
        if karg:
            raise ValueError('Index replace error: it got unexpected field names: %r'%karg.keys())
        result.__dict__=OrderedDict()
        for key,value in zip(self.__dict__.keys(),result):
            result.__dict__[key]=value
        return result

class Internal(object):
    '''
    This class is the base class for all internal degrees of freedom in a point.
    '''

ANNIHILATION,CREATION=0,1

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
    This class defines the fermionic degrees of freedom in a point.
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
                A number to indicate whether or not the Nambu space is used. 1 means no and 2 means yes.
        '''
        self.atom=atom
        self.norbital=norbital
        self.nspin=nspin
        self.nnambu=nnmabu

    def __str__(self):
        '''
        Convert an instance to string.
        '''
        return 'Atom,norbital,nspin,nnambu: '+str(self.atom)+', '+str(self.norbital)+', '+str(self.nspin)+', '+str(self.nnambu)

    def __eq__(self,other):
        '''
        Overloaded operator(==).
        '''
        return self.atom==other.atom and self.norbital==other.norbital and self.nspin==other.nspin and self.nnambu==other.nnambu

    def table(self,pid,nambu=False,key=None):
        '''
        This method returns a Table instance that contains all the allowed indices.
        Parameters:
            pid: PID
                The spatial part of the indices.
            nambu: logical, optional
                A flag to tag whether or not the nambu space is used.
            key: function
                The key function used to sort the indices.
        Returns:
            result: Table
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
        if nambu in (0,1):
            return fid.orbital+fid.spin*self.norbital+fid.nambu*self.norbital*self.nspin
        else:
            raise ValueError("Point seq_state error: the nambu index must be 0 or 1.")

    def state_index(self,seq_state):
        '''
        This methods returns the the orbital, spin and nambu index of a state whose sequence equals the input seq_state.
        Parameters:
            seq_state: integer
                The sequence of the state.
        Returns:
            A dict in the form {'spin':...,'orbital':...,'nambu':...}
        Note: This method should be used in pairs with the method seq_state to ensure the correct sequence-index correspondence.
        '''
        spin=seq_state%(self.norbital*self.nspin)/self.norbital
        orbital=seq_state%(self.norbital*self.nspin)%self.norbital
        nambu=seq_state/(self.norbital*self.nspin)
        return FID(spin=spin,orbital=orbital,nambu=nambu)

class Configuration(dict):
    '''
    '''
    def __init__(self,dict=None,priority=None):
        '''
        Constructor.
        '''
        if dict is not None:
            self.update(dict)
        self.priority=priority

    def table(self):
        pass
