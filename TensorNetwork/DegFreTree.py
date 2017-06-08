'''
===================================
Tree of internal degrees of freedom
===================================
The tree of physical degrees of freedom, including:
    * constants: DEGFRE_FERMIONIC_PRIORITY,DEGFRE_FERMIONIC_LAYERS,DEGFRE_SPIN_PRIORITY,DEGFRE_SPIN_LAYERS
    * classes: DegFreTree
'''

__all__=['DEGFRE_FERMIONIC_PRIORITY','DEGFRE_SPIN_PRIORITY','DEGFRE_FERMIONIC_LAYERS','DEGFRE_SPIN_LAYERS','DegFreTree']

import numpy as np
from HamiltonianPy import PID,Table,QuantumNumbers
from ..Misc import Tree
from Tensor import Label
from collections import OrderedDict

DEGFRE_FERMIONIC_PRIORITY=('scope','site','orbital','spin','nambu')
DEGFRE_FERMIONIC_LAYERS=[('scope','site','orbital'),('spin',)]
DEGFRE_SPIN_PRIORITY=['scope','site','orbital','S']
DEGFRE_SPIN_LAYERS=[('scope','site','orbital','S')]

class DegFreTree(Tree):
    '''
    The tree of the layered degrees of freedom.
    For each (node,data) pair of the tree,
        * node: Index
            The selected index which can represent a couple of indices.
        * data: integer or QuantumNumbers
            When an integer, it is the number of degrees of freedom that the index represents;
            When a QuantumNumbers, it is the quantum number collection that the index is associated with.
    Attributes
    ----------
    mode : 'QN' or 'NB'
        The mode of the DegFreTree.
    layers : list of tuples of string
        The tag of each layer of indices.
    priority : lsit of string
        The sequence priority of the allowed indices.
    map : function
        This function maps a leaf (bottom index) of the DegFreTree to its corresponding data.
    cache : dict
        The cache of the degfretree.
    '''

    def __init__(self,mode,layers,priority,leaves=[],map=None):
        '''
        Constructor.
        Parameters
        ----------
        mode : 'QN' or 'NB'
            The mode of the DegFreTree.
        layers : list of tuples of string
            The tag of each layer of indices.
        priority : lsit of string
            The sequence priority of the allowed indices.
        leaves : list of Index, optional
            The leaves (bottom indices) of the DegFreTree.
        map : function, optional
            This function maps a leaf (bottom index) of the DegFreTree to its corresponding data.
        '''
        self.reset(mode=mode,layers=layers,priority=priority,leaves=leaves,map=map)

    def reset(self,mode=None,layers=None,priority=None,leaves=[],map=None):
        '''
        Reset the DegFreTree.
        Parameters
        ----------
        mode,layers,priority,leaves,map :
            Please see DegFreTree.__init__ for details.
        '''
        self.clear()
        Tree.__init__(self)
        assert mode in (None,'QN','NB')
        if mode is not None: self.mode=mode
        if layers is not None: self.layers=layers
        if priority is not None: self.priority=priority
        if map is not None: self.map=map
        self.cache={}
        if len(leaves)>0:
            temp=[key for layer in self.layers for key in layer]
            assert set(range(len(PID._fields)))==set([temp.index(key) for key in PID._fields])
            temp=set(temp)
            self.add_leaf(parent=None,leaf=tuple([None]*len(leaves[0])),data=None)
            for layer in self.layers:
                temp-=set(layer)
                indices=set([index.replace(**{key:None for key in temp}) for index in leaves])
                self.cache[('indices',layer)]=sorted(indices,key=lambda index: index.to_tuple(priority=self.priority))
                self.cache[('table',layer)]=Table(self.indices(layer=layer))
                for index in self.indices(layer=layer):
                    self.add_leaf(parent=index.replace(**{key:None for key in layer}),leaf=index,data=None)
            for i,layer in enumerate(reversed(self.layers)):
                if i==0:
                    for index in self.indices(layer):
                        self[index]=self.map(index)
                else:
                    for index in self.indices(layer):
                        if self.mode=='QN':
                            self[index]=QuantumNumbers.kron([self[child] for child in self.children(index)])
                        else:
                            self[index]=np.product([self[child] for child in self.children(index)])

    def ndegfre(self,index):
        '''
        The number of degrees of freedom represented by index.
        Parameters
        ----------
        index : Index
            The index of the degrees of freedom.
        Returns
        -------
        integer
            The number of degrees of freedom.
        '''
        if self.mode=='NB':
            return self[index]
        else:
            return len(self[index])

    def indices(self,layer=0):
        '''
        The indices in a layer.
        Parameters
        ----------
        layer : integer/tuple-of-string, optional
            The layer where the indices are restricted.
        Returns
        -------
        list of Index
            The indices in the requested layer.
        '''
        return self.cache[('indices',self.layers[layer] if type(layer) in (int,long) else layer)]

    def table(self,layer=0):
        '''
        Return a index-sequence table with the index restricted on a specific layer.
        Parameters
        ----------
        layer : integer/tuple-of-string
            The layer where the indices are restricted.
        Returns
        -------
        Table
            The index-sequence table.
        '''
        return self.cache[('table',self.layers[layer] if type(layer) in (int,long) else layer)]

    def labels(self,mode,layer=0):
        '''
        Return the inquired labels.
        Parameters
        ----------
        mode : 'B','S','O'
            * 'B' for bond labels of an mps;
            * 'S' for site labels of an mps or an mpo;
            * 'O' for bond labels of an mpo.
        layer : integer/tuple-of-string, optional
            The layer information of the inquired labels.
        Returns
        -------
        list of Label
            The inquired labels.
        '''
        mode,layer=mode.upper(),self.layers[layer] if type(layer) in (int,long) else layer
        assert mode in ('B','S','O')
        if ('labels',mode,layer) not in self.cache:
            if mode in ('B','O'):
                result=[Label(identifier='%s%s-%s'%(mode,self.layers.index(layer),i),qns=None) for i in xrange(len(self.indices(layer))+1)]
            else:
                result=[Label(identifier=index,qns=self[index]) for index in self.indices(layer)]
            self.cache[('labels',mode,layer)]=result
        return self.cache[('labels',mode,layer)]
