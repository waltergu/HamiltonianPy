'''
The inner structure of a point.
'''
from IndexPy import *
from TablePy import *
class Struct(object):
    '''
    This class is the base class for all inner stuctures of a point.
    '''
    def __init__(self,**karg):
        '''
        Constructor.
        '''
        for key in karg:
            setattr(self,key,karg[key])

    def __ne__(self,other):
        '''
        Overloaded operator(!=).
        '''
        return not self==other

class Fermi(Struct):
    '''
    This class defines the fermi structure of a point.
    Attributes:
        atom: integer, default value 0
            The atom species on this point.
        norbital: integer, default value 1
            Number of orbitals.
        nspin: integer, default value 2
            Number of spins.
        nnambu: integer, default value 1.
            An integer to indicate whether or not using the Nambu space. 1 means no and 2 means yes.
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
        super(Fermi,self).__init__(atom=atom,norbital=norbital,nspin=nspin,nnambu=nnambu)

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

    def table(self,scope,site,nambu=False,priority=None):
        '''
        This method returns a Table instance that contains all the allowed indices which can be defined on this structure.
        Parameters:
            scope: string
                The scope of the index.
            site: integer
                The site index.
            nambu: logical, optional
                A flag to tag whether or not the nambu space is used.
            priority: function
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
                        result.append(Index(scope=scope,site=site,orbital=orbital,spin=spin,nambu=buff))
        else:
            for spin in xrange(self.nspin):
                for orbital in xrange(self.norbital):
                    result.append(Index(scope=scope,site=site,orbital=orbital,spin=spin,nambu=ANNIHILATION))
        if priority is None:
            return Table(result)
        else:
            return Table(sorted(result,key=priority))


    def seq_state(self,orbital,spin,nambu):
        '''
        This methods is the oversimplified version of returning the sequence of a input state with orbital, spin and nambu index assigned.
        Note: the priority to generate the sequence cannot be modified by the users and is always "NSO".
        '''
        if nambu in (0,1):
            return orbital+spin*self.norbital+nambu*self.norbital*self.nspin
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
        return {'spin':spin,'orbital':orbital,'nambu':nambu}
