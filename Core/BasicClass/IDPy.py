'''
ID.
'''

__all__=['ID']

class ID:
    '''
    This class provides a general hashable id for an object.
    It can have any attribute by the key work argument construction.
    '''
    def __init__(self,**karg):
        '''
        Constructor.
        '''
        for key,value in karg.iteritems():
            setattr(self,key,value)

    def __repr__(self):
        '''
        Convert an instance to string.
        '''
        return '<ID>'+self.__str__()

    def __str__(self):
        '''
        Convert an instance to string.
        '''
        buff=[]
        for key,value in self.__dict__.iteritems():
            buff.append('%s:%s'%(key,value))
        return '('+', '.join(buff)+')'

    def __hash__(self):
        '''
        Give an ID instance a Hash value.
        '''
        return hash(str(self.__dict__))

    def __eq__(self,other):
        '''
        Overloaded operator(==).
        '''
        if str(self.__dict__)==str(other.__dict__):
            return True
        else:
            return False
            
    def __ne__(self,other):
        '''
        Overloaded operator(!=).
        '''
        return not self==other
