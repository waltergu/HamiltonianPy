'''
Index.
'''
from copy import deepcopy
from re import split
ANNIHILATION=0
CREATION=1

class Index:
    '''
    This class provides a linear operator with an index.
    Attributes:
        site: integer
            The site index, start with 0, default value 0.
        orbital: integer
            The orbital index, start with 0, default value 0. 
        spin: integer
            The spin index, start with 0, default value 0.
        nambu: integer
            '0' for ANNIHILATION and '1' for CREATION, default value ANNIHILATION.
        scope: string
            The scope of the index within which it is defined, default value 'None'.
    '''
    
    def __init__(self,site=0,orbital=0,spin=0,nambu=ANNIHILATION,scope=None):
        self.site=site
        self.orbital=orbital
        self.spin=spin
        self.nambu=nambu
        self.scope=str(scope)
    
    def __str__(self):
        '''
        Convert an instance to string.
        '''
        if self.scope=='None':
            return 'Site,orbital,spin,nambu: '+str(self.site)+', '+str(self.orbital)+', '+str(self.spin)+', '+('ANNIHILATION' if self.nambu==ANNIHILATION else 'CREATION')+'\n'
        else: 
            return 'Scope,site,orbital,spin,nambu: '+self.scope+', '+str(self.site)+', '+str(self.orbital)+', '+str(self.spin)+', '+('ANNIHILATION' if self.nambu==ANNIHILATION else 'CREATION')+'\n'

    def __repr__(self):
        '''
        Convert an instance to string in a formal environment.
        '''
        return self.scope+str(self.site)+str(self.orbital)+str(self.spin)+str(self.nambu)
    
    def __hash__(self):
        '''
        Give an Index instance a Hash value.
        '''
        return hash(self.scope+str(self.site)+str(self.orbital)+str(self.spin)+str(self.nambu))
        
    def __eq__(self,other):
        '''
        Overloaded operator(==).
        '''
        if self.scope==other.scope and self.site==other.site and self.orbital==other.orbital and self.spin==other.spin and self.nambu==other.nambu:
            return True
        else:
            return False
            
    def __ne__(self,other):
        '''
        Overloaded operator(!=).
        '''
        return not self==other
    
    @property
    def dagger(self):
        '''
        The dagger of the index, which keeps site, orbital and spin unchanged while sets nambu from CREATION to ANNIHILATION or vice versa. 
        '''
        return Index(self.site,self.orbital,self.spin,1-self.nambu,self.scope)

    def to_str(self,indication):
        '''
        Convert an instance to string according to the parameter indication.
        '''
        result=''
        for i in indication:
            if i in ('N','n'):
                result+=str(self.nambu)+' '
            elif i in ('S','s'):
                result+=str(self.spin)+' '
            elif i in ('C','c'):
                result+=str(self.site)+' '
            elif i in ('O','o'):
                result+=str(self.orbital)+' '
            elif i in ('P','p'):
                result+=self.scope+' '
        return result

    def to_tuple(self,indication):
        '''
        Convert an instance to tuple according to the parameter indication.
        '''
        result=()
        for i in indication:
            if i in ('N','n'):
                result+=(self.nambu,)
            elif i in ('S','s'):
                result+=(self.spin,)
            elif i in ('C','c'):
                result+=(self.site,)
            elif i in ('O','o'):
                result+=(self.orbital,)
            elif i in ('P','p'):
                result+=(self.scope,)
        return result

def to_index(str,indication):
    '''
    Convert a string to index according to the parameter indication.
    '''
    result=Index()
    values=split('\W+',str)
    for (i,v),value in zip(enumerate(indication),values):
        if v in ('N','n'):
            result.nambu=int(value)
        elif v in ('S','s'):
            result.spin=int(value)
        elif v in ('C','c'):
            result.site=int(value)
        elif v in ('O','o'):
            result.orbital=int(value)
        elif v in ('P','p'):
            result.scope=value
        else:
            raise ValueError('To_index error: each element of the indication must be "N" or "n"(nambu), "S" or "s"(spin), "C" or "c"(site), "O" or "o"(orbital), "P" or "p"(scope).')
    return result
