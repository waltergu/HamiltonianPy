'''
The tree of physical degrees of freedom, including:
1) constants: DEFAULT_FERMIONIC_LAYERS,DEFAULT_SPIN_LAYERS
2) classes: DegFreTree
'''

__all__=['DEFAULT_FERMIONIC_LAYERS','DEFAULT_SPIN_LAYERS','DegFreTree']

import numpy as np
from HamiltonianPy import PID,Table,QuantumNumbers
from ..Misc import Tree
from Tensor import Label
from collections import OrderedDict

DEFAULT_FERMIONIC_LAYERS=[('scope',),('site',),('orbital','spin')]
DEFAULT_SPIN_LAYERS=[('scope',),('site','S')]

class DegFreTree(Tree):
    '''
    The tree of the layered degrees of freedom.
    For each (node,data) pair of the tree,
        node: Index
            The selected index which can represent a couple of indices.
        data: integer of QuantumNumbers
            When an integer, it is the number of degrees of freedom that the index represents;
            When a QuantumNumbers, it is the quantum number collection that the index is associated with.
    Attributes:
        mode: 'QN' or 'NB'
            The mode of the DegFreTree.
        layers: list of tuples of string
            The tag of each layer of indices.
        priority: lsit of string
            The sequence priority of the allowed indices.
        map: function
            This function maps a leaf (bottom index) of the DegFreTree to its corresponding data.
        cache: dict
            The cache of the degfretree.
    '''

    def __init__(self,mode,layers,priority,leaves=[],map=None):
        '''
        Constructor.
        Parameters:
            mode: 'QN' or 'NB'
                The mode of the DegFreTree.
            layers: list of tuples of string
                The tag of each layer of indices.
            priority: lsit of string
                The sequence priority of the allowed indices.
            leaves: list of Index, optional
                The leaves (bottom indices) of the DegFreTree.
            map: function, optional
                This function maps a leaf (bottom index) of the DegFreTree to its corresponding data.
        '''
        self.reset(mode=mode,layers=layers,priority=priority,leaves=leaves,map=map)

    def reset(self,mode=None,layers=None,priority=None,leaves=[],map=None):
        '''
        Reset the DegFreTree.
        Parameters:
            mode,layers,priority,leaves,map: optional
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
                leaves,permutations,antipermutations=[],[],[]
                if i==0:
                    for index in self.indices(layer):
                        self[index]=self.map(index)
                        leaves.append([index])
                        permutations.append(None)
                        antipermutations.append(None)
                else:
                    lowertable=self.cache[('table',self.layers[-i])]
                    lowerleaves=self.cache[('leaves',self.layers[-i])]
                    for index in self.indices(layer):
                        leaves.append([leaf for child in self.children(index) for leaf in lowerleaves[lowertable[child]]])
                        if self.mode=='QN':
                            qns,permutation=QuantumNumbers.kron([self[leaf] for leaf in leaves[-1]]).sort(history=True)
                        else:
                            qns,permutation=np.product([self[leaf] for leaf in leaves[-1]]),None
                        self[index]=qns
                        permutations.append(permutation)
                        antipermutations.append(None if permutation is None else np.argsort(permutation))
                self.cache[('leaves',layer)]=leaves
                self.cache[('permutations',layer)]=permutations
                self.cache[('antipermutations',layer)]=antipermutations


    def ndegfre(self,index):
        '''
        The number of degrees of freedom represented by index.
        Parameters:
            index: Index
                The index of the degrees of freedom.
        Returns: integer
            The number of degrees of freedom.
        '''
        if self.mode=='NB':
            return self[index]
        else:
            return len(self[index])

    def indices(self,layer):
        '''
        The indices in a layer.
        Parameters:
            layer: tuple of string
                The layer where the indices are restricted.
        Returns: list of Index
            The indices in the requested layer.
        '''
        return self.cache[('indices',layer)]

    def table(self,layer):
        '''
        Return a index-sequence table with the index restricted on a specific layer.
        Parameters:
            layer: tuple of string
                The layer where the indices are restricted.
        Returns: Table
            The index-sequence table.
        '''
        return self.cache[('table',layer)]

    def leaves(self,layer):
        '''
        Return a list in the form [leaves]
            leaves: list of Index
                The leaves of a branch on a specific layer.
        Parameters:
            layer: tuple of string
                The layer where the branches are restricted.
        Returns: as above.
        '''
        return self.cache[('leaves',layer)]

    def permutations(self,layer):
        '''
        Return a list in the form [permutation]
            permutation: 1d ndarray
                The permutation array of the quantum numbers of a branch on a specific layer.
        Parameters:
            layer: tuple of string
                The layer where the branches are restricted.
        Returns: as above.
        '''
        return self.cache[('permutations',layer)]

    def antipermutations(self,layer):
        '''
        Return a list in the form [antipermutation]
            antipermutation: 1d ndarray
                The antipermutation array of the quantum numbers of a branch on a specific layer.
        Parameters:
            layer: tuple of string
                The layer where the branches are restricted.
        Returns: as above.
        '''
        return self.cache[('antipermutations',layer)]

    def labels(self,layer,mode):
        '''
        Return the inquired labels.
        Parameters:
            layer: tuple of string
                The layer information of the inquired labels.
            mode: 'B','S','O','M'
                'B' for bond labels of an mps;
                'S' for site labels of an mps or an mpo;
                'O' for bond labels of an mpo.
                'M' for bottom labels of the physical degrees of freedom of an mps or an mpo.
        Returns: list of Label
                The inquired labels.
        '''
        mode=mode.upper()
        assert mode in ('B','S','O','M')
        if ('labels',layer,mode) not in self.cache:
            if mode in ('B','O'):
                result=[Label(identifier='%s%s-%s'%(mode,self.layers.index(layer),i),qns=None) for i in xrange(len(self.indices(layer))+1)]
            elif mode=='S':
                result=[Label(identifier=index,qns=self[index]) for index in self.indices(layer)]
            else:
                result=[[Label(leaf,qns=self[leaf]) for leaf in leaves] for leaves in self.leaves(layer)]
            self.cache[('labels',layer,mode)]=result
        return self.cache[('labels',layer,mode)]
