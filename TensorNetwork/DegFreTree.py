'''
The tree of physical degrees of freedom, including:
1) constants: DEFAULT_FERMIONIC_LAYERS,DEFAULT_SPIN_LAYERS
2) classes: DegFreTree
'''

__all__=['DEFAULT_FERMIONIC_LAYERS','DEFAULT_SPIN_LAYERS','DegFreTree']

import numpy as np
from HamiltonianPy import PID,Table
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
        data: integer of QuantumNumberCollection
            When an integer, it is the number of degrees of freedom that the index represents;
            When a QuantumNumberCollection, it is the quantum number collection that the index is associated with.
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
            mode: 'QN' or 'NB', optional
                The mode of the DegFreTree.
            layers: list of tuples of string, optional
                The tag of each layer of indices.
            priority: lsit of string, optional
                The sequence priority of the allowed indices.
            leaves: list of Index, optional
                The leaves (bottom indices) of the DegFreTree.
            map: function, optional
                This function maps a leaf (bottom index) of the DegFreTree to its corresponding data.
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
                self.cache[('reversed_table',layer)]=self.table(layer=layer).reversed_table
                for index in self.indices(layer=layer):
                    self.add_leaf(parent=index.replace(**{key:None for key in layer}),leaf=index,data=None)
            if self.mode=='QN':
                for i,layer in enumerate(reversed(self.layers)):
                    key=('qnc_evolutions',layer)
                    self.cache[key]={}
                    if i==0:
                        for index in self.indices(layer):
                            self[index]=self.map(index)
                            self.cache[key][index]=[self[index]]
                    else:
                        for index in self.indices(layer):
                            buff=[]
                            for j,child in enumerate(self.children(index)):
                                if j==0:
                                    buff.append(self[child])
                                else:
                                    buff.append(buff[j-1].kron(self[child],'+',history=True))
                            self.cache[key][index]=buff
                            self[index]=buff[-1]
            else:
                for i,layer in enumerate(reversed(self.layers)):
                   for index in self.indices(layer):
                        self[index]=self.map(index) if i==0 else np.product([self[child] for child in self.children(index)])

    def ndegfre(self,index):
        '''
        The number of degrees of freedom reprented by index.
        Parameters:
            index: Index
                The index of the degrees of freedom.
        Returns: integer
            The number of degrees of freedom.
        '''
        if self.mode=='NB':
            return self[index]
        else:
            return self[index].n

    def indices(self,layer=None):
        '''
        The indices in a layer.
        Parameters:
            layer: tuple of string
                The layer where the indices are restricted.
        Returns: list of Index
            The indices in the requested layer.
        '''
        layer=self.layers[0] if layer is None else layer
        return self.cache[('indices',layer)]

    def table(self,layer=None):
        '''
        Return a index-sequence table with the index restricted on a specific layer.
        Parameters:
            layer: tuple of string
                The layer where the indices are restricted.
        Returns: Table
            The index-sequence table.
        '''
        layer=self.layers[0] if layer is None else layer
        return self.cache[('table',layer)]

    def reversed_table(self,layer=None):
        '''
        Return the reversed sequence-index table with the index restricted on a specific layer.
        Parameters:
            layer: tuple of string
                The layer where the indices are restricted.
        Returns: Table
            The sequence-index table.
        '''
        layer=self.layers[0] if layer is None else layer
        return self.cache[('reversed_table',layer)]

    def qnc_evolutions(self,layer=None):
        '''
        Retrun a dict in the form {branch:qncs}
            branch: index
                A branch on a specific layer.
            qncs: list of QuantumNumberCollection
                The kron path of the quantum number collection corresponding to the branch.
        Parameters:
            layer: tuple of string
                The layer where the branches are restricted.
        Returns: as above.
        '''
        layer=self.layers[0] if layer is None else layer
        return self.cache.get(('qnc_evolutions',layer),None)

    def labels(self,layer=None,full_labels=True):
        '''
        Return a OrderedDict of labels on a specific layer.
        Parameters:
            layer: tuple of string
                The layer where the labels are restricted.
            full_labels: logical, optional
                When True the full labels including the bond labels and physical labels will be returned;
                When Flase, only the physical labels will be returned.
        Returns: 
            When full_labels is True: OrderedDict of 3-tuple (L,S,R)
            When full_labels is False: OrderedDict of Label S
            L,S,R are in the following form:
                L=Label(identifier=i,qnc=None)
                S=Label(identifier=index,qnc=self[index])
                R=Label(identifier=(i+1)%len(indices),qnc=None)
            with,
                identifier: Index
                    The physical degrees of freedom of the labels restricted on this layer.
                i: integer
                    The sequence of the labels restricted on this layer.
                N: integer or QuantumNumberCollection
                    When self.mode=='QN', it is a QuantumNumberCollection, the quantum number collection of the physical degrees of freedom;
                    When self.mode=='NB', it is a integer, the number of the physical degrees of freedom.
        '''
        layer=self.layers[0] if layer is None else layer
        if ('labels',layer,full_labels) not in self.cache:
            result=OrderedDict()
            for i,index in enumerate(self.indices(layer)):
                S=Label(identifier=index,qnc=self[index])
                if full_labels:
                    L=Label(identifier=i,qnc=None)
                    R=Label(identifier=(i+1)%len(self.indices(layer)),qnc=None)
                    result[index]=(L,S,R)
                else:
                    result[index]=S
            self.cache[('labels',layer,full_labels)]=result
        return self.cache[('labels',layer,full_labels)]
