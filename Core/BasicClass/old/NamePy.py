'''
Name of Engine.
'''
from collections import OrderedDict
class Name:
    '''
    This class provides an engine with a name.
    Attributes:
        prefix: string
            Description of the engine.
        suffix: string
            Additional remarks of the engine.
        _alter: OrderedDict
            It contains the contant parameters of the engine.
        _const: OrderedDict
            It contains the alterable parameters of the engine.
    '''
    
    def __init__(self,prefix='',suffix=''):
        self.prefix=prefix
        self.suffix=suffix
        self._const=OrderedDict()
        self._alter=OrderedDict()
    
    def __str__(self):
        return self.full

    def update(self,const=None,alter=None):
        if const is not None:
            self._const.update(const)
        if alter is not None:
            self._alter.update(alter)

    @property
    def const(self):
        '''
        This method returns a string containing only contant parameters as the name of the engine.
        '''
        result=self.prefix+'_'
        for obj in self._const.itervalues():
            result+=repr(obj)+'_'
        result+=self.suffix
        return result

    @property
    def alter(self):
        '''
        This method returns a string containing only alterable parameters as the name of the engine.
        '''
        result=self.prefix+'_'
        for obj in self._alter.itervalues():
            result+=repr(obj)+'_'
        result+=self.suffix
        return result

    @property
    def full(self):
        '''
        This method returns a string containing both contant parameters and alterable parameters as the name of the engine.
        '''
        result=self.prefix+'_'
        for obj in self._const.itervalues():
            result+=repr(obj)+'_'
        for obj in self._alter.itervalues():
            result+=repr(obj)+'_'
        result+=self.suffix
        return result
