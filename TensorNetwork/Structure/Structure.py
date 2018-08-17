'''
================================
The structure of tensor networks
================================

The structure of tensor networks, including:
    * constants: DEGFRE_FOCK_LAYERS, DEGFRE_FERMIONIC_LAYERS, DEGFRE_BOSONIC_LAYERS, DEGFRE_SPIN_LAYERS
    * classes: DegFreTree
'''

__all__=['DEGFRE_FOCK_LAYERS','DEGFRE_FERMIONIC_LAYERS','DEGFRE_BOSONIC_LAYERS','DEGFRE_SPIN_LAYERS','DegFreTree']

import numpy as np
from HamiltonianPy import PID,Table,QuantumNumbers
from HamiltonianPy.Misc import Tree
from HamiltonianPy.TensorNetwork.Tensor import Label
from collections import OrderedDict

DEGFRE_FOCK_LAYERS=[('scope','site','orbital'),('spin',)]
DEGFRE_FERMIONIC_LAYERS=[('scope','site','orbital'),('spin',)]
DEGFRE_BOSONIC_LAYERS=[('scope','site','orbital'),('spin',)]
DEGFRE_SPIN_LAYERS=[('scope','site','orbital','S')]

class DegFreTree(Tree):
    '''
    The tree of the layered degrees of freedom.
    For each (node,data) pair of the tree,
        * node : Index
            The selected index which can represent a couple of indices.
        * data : int or QuantumNumbers
            When an int, it is the number of degrees of freedom that the index represents;
            When a QuantumNumbers, it is the quantum number collection that the index is associated with.

    Attributes
    ----------
    layers : list of tuples of str
        The tag of each layer of indices.
    map : callable
        This function maps a leaf (bottom index) of the DegFreTree to its corresponding data.
    cache : dict
        The cache of the degfretree.
    '''

    def __init__(self,layers,leaves=(),map=None):
        '''
        Constructor.

        Parameters
        ----------
        layers : list of tuples of str
            The tag of each layer of indices.
        leaves : list of Index, optional
            The leaves (bottom indices) of the DegFreTree.
        map : callable, optional
            This function maps a leaf (bottom index) of the DegFreTree to its corresponding data.
        '''
        self.reset(layers=layers,leaves=leaves,map=map)

    @property
    def dtype(self):
        '''
        The type of the data of the degfretree.
        '''
        return type(self[self.indices(-1)[0]]) if len(self)>0 else None

    def reset(self,layers=None,leaves=(),map=None):
        '''
        Reset the DegFreTree.

        Parameters
        ----------
        layers,leaves,map :
            Please see DegFreTree.__init__ for details.
        '''
        self.clear()
        Tree.__init__(self)
        if layers is not None: self.layers=layers
        if map is not None: self.map=map
        self.cache={}
        if len(leaves)>0:
            temp=[key for layer in self.layers for key in layer]
            assert set(range(len(PID._fields)))==set([temp.index(key) for key in PID._fields])
            temp=set(temp)
            self.addleaf(parent=None,leaf=tuple([None]*len(leaves[0])),data=None)
            for layer in self.layers:
                temp-=set(layer)
                self.cache[('indices',layer)]=list(OrderedDict([(index.replace(**{key:None for key in temp}),None) for index in leaves]).keys())
                self.cache[('table',layer)]=Table(self.indices(layer=layer))
                for index in self.indices(layer=layer):
                    self.addleaf(parent=index.replace(**{key:None for key in layer}),leaf=index,data=None)
            for i,layer in enumerate(reversed(self.layers)):
                if i==0:
                    for index in self.indices(layer): self[index]=self.map(index)
                else:
                    dtype=self.dtype
                    for index in self.indices(layer):
                        self[index]=(QuantumNumbers.kron if issubclass(dtype,QuantumNumbers) else np.product)([self[child] for child in self.children(index)])

    def ndegfre(self,index):
        '''
        The number of degrees of freedom represented by index.

        Parameters
        ----------
        index : Index
            The index of the degrees of freedom.

        Returns
        -------
        int
            The number of degrees of freedom.
        '''
        qns=self[index]
        return len(qns) if isinstance(qns,QuantumNumbers) else qns

    def indices(self,layer=0):
        '''
        The indices in a layer.

        Parameters
        ----------
        layer : int/tuple-of-str, optional
            The layer where the indices are restricted.

        Returns
        -------
        list of Index
            The indices in the requested layer.
        '''
        return self.cache[('indices',self.layers[layer] if type(layer) is int else layer)]

    def table(self,layer=0):
        '''
        Return a index-sequence table with the index restricted on a specific layer.

        Parameters
        ----------
        layer : int/tuple-of-str
            The layer where the indices are restricted.

        Returns
        -------
        Table
            The index-sequence table.
        '''
        return self.cache[('table',self.layers[layer] if type(layer) is int else layer)]

    def labels(self,mode,layer=0):
        '''
        Return the inquired labels/identifiers.

        Parameters
        ----------
        mode : 'B','S','O'
            * 'B' for bond labels of an mps;
            * 'S' for site labels of an mps or an mpo;
            * 'O' for bond labels of an mpo.
        layer : int/tuple-of-str, optional
            The layer information of the inquired labels.

        Returns
        -------
        list of Label/str
            The inquired labels/identifiers.
        '''
        mode,layer=mode.upper(),self.layers[layer] if type(layer) is int else layer
        assert mode in ('B','S','O')
        if ('labels',mode,layer) not in self.cache:
            if mode in ('B','O'):
                result=['%s%s-%s'%(mode,self.layers.index(layer),i) for i in range(len(self.indices(layer))+1)]
            else:
                result=[Label(index,self[index],None) for index in self.indices(layer)]
            self.cache[('labels',mode,layer)]=result
        return self.cache[('labels',mode,layer)]
