'''
Tree structure, including:
1) classes: Tree
'''

__all__=['Tree']

class Tree(dict):
    '''
    Tree class.
    Attributes:
        root: hashable object
            The Tree's root node.
        _parent: dict
            The lookup dict for parents.
        _children: dict
            The lookup dict for children.
    '''

    ROOT=None
    (DEPTH,WIDTH)=list(range(2))
    (NODE,DATA,PAIR)=list(range(3))

    def __init__(self,root=None,data=None):
        '''
        Constructor.
        Parameters:
            root: hashable object, optional
                The Tree's root node.
            data: any object, optional
                The data of the root node.
        '''
        self.root=None
        self._parent={}
        self._children={}
        if root is not None:
            self.add_leaf(None,root,data)

    def add_leaf(self,parent,leaf,data=None):
        '''
        Add a leaf for the tree.
        Parameters:
            parent: hashable object
                The parent of the new leaf.
            leaf: hashable object
                The new leaf.
            data: any object, optional
                The data of the new leaf.
        '''
        if self.root is None:
            if parent is None:
                self.root=leaf
            else:
                raise ValueError('Tree add_leaf error: the parent for the first leaf of the tree must be None.')
        elif parent in self:
            self._children[parent].append(leaf)
        else:
            raise ValueError('Tree add_leaf error: the parent of the leaf does not exist.')
        self._parent[leaf]=parent
        super(Tree,self).__setitem__(leaf,data)
        self._children[leaf]=[]

    def __setitem__(self,node,data):
        '''
        Set the data of an existing node.
        Parameters:
            node: hashable object
                The node.
            para: any object
                The data of the node.
        '''
        if node in self:
            dict.__setitem__(self,node,data)
        else:
            raise ValueError('Tree __setitem__ error: the node of the tree does not exist.')

    def add_subtree(self,subtree,parent=None):
        '''
        Add a subtree.
        Parameters:
            subtree: Tree
                The subtree to be added.
            parent: hashable object
                The parent of the subtree's root.
        '''
        if isinstance(subtree,Tree):
            if self.root is None:
                self.root=subtree.root
                self.update(subtree)
                self._parent.update(subtree._parent)
                self._children.update(subtree._children)
            elif parent in self:
                if len(set(subtree.keys())&set(self.keys()))==0:
                    self.update(subtree)
                    self._parent.update(subtree._parent)
                    self._children.update(subtree._children)
                    self._parent[subtree.root]=parent
                    self._children[parent].append(subtree.root)
                else:
                    raise ValueError('Tree add_subtree error: the subtree to be added cannot share any same node with the parent tree.')
            else:
                raise ValueError('Tree add_subtree error: the parent of the subtree does not exist.')
        else:
            raise ValueError('Tree add_subtree error: the first parameter must be an instance of Tree.')

    def remove_subtree(self,node):
        '''
        Remove a subtree.
        Parameters:
            node: hashable object
                The root node of the to-be-removed subtree.
        '''
        if node==self.root:
            self.root=None
            for key in self.keys():
                self.pop(key,None)
                self._parent.pop(key,None)
                self._children.pop(key,None)
        else:
            self._children[self._parent[node]].remove(node)
            del self._parent[node]
            temp=[]
            for key in self.expand(node=node,mode=Tree.DEPTH,return_form=Tree.NODE):
                temp.append(key)
            for key in temp:
                self.pop(key,None)
                self._children.pop(key,None)

    def move_subtree(self,node,parent):
        '''
        Move a subtree to a child of a new parent.
        Parameters:
            node: hashable object
                The root node of the to-be-moved subtree.
            parent: hashable object
                The new parent of the the subtree.
        '''
        self._children[self._parent[node]].remove(node)
        self._parent[node]=parent
        self._children[parent].append(node)

    def parent(self,node):
        '''
        Return the parent of a node.
        Parameters:
            node: hashable object
                The node whose parent is inquired.
        Returns: hashable object
            The parent.
        '''
        return self._parent[node]

    def children(self,node):
        '''
        Return a list of a node's children.
        Parameters:
            node: hashable object
                The node whose children are inquired.
        Returns: list of hashable object
            The list of the children.
        '''
        return self._children[node]

    def siblings(self,node):
        '''
        Return a list of the siblings (i.e. the nodes sharing the same parent) of a node.
        Parameters:
            node: hashable object
                The node whose siblings are inquired.
        Returns: list of hashable object
            The list of the siblings.
        '''
        if node!=self.root:
            return [key for key in self._children[self._parent[node]] if key!=node]
        else:
            return []

    def subtree(self,node):
        '''
        Return a subtree whose root is node.
        Parameters:
            node: hashable object
                The root node of the subtree.
        Returns: Tree
            The subtree.
        '''
        result=Tree()
        result.root=node
        for node,data in self.expand(node=node,return_form=Tree.PAIR):
            super(Tree,result).__setitem__(node,data)
            result._parent[node]=self._parent[node]
            result._children[node]=self._children[node]
        return result

    def is_leaf(self,node):
        '''
        Judge whether a node is a leaf (a node without children) or not.
        Parameters:
            node: any hashable object
                The node to be judged.
        Returns: logical
            True for is False for not.
        '''
        return len(self._children[node])==0

    @property
    def leaves(self):
        '''
        Return all the leaves contained in this tree.
        '''
        return [node for node in self if self.is_leaf(node)]

    def expand(self,node=None,mode=DEPTH,return_form=PAIR):
        '''
        Expand the Tree.
        Parameters:
            node: hashable object,optional
                The node with which to begin to expand the Tree.
                If it is None, the expansion begins with the root by default.
            mode: Tree.DEPTH/Tree.WIDTH, optional
                The flag to choose depth-first expansion or width-first expansion.
            return_form: Tree.PAIR/Tree.NODE/Tree.DATA, optional
                The flag to set the form of the returned iterator.
        Returns: iterator
            if return_form==Tree.PAIR, the returned iterator runs over tuples in the form (node,data);
            if return_form==Tree.DATA, the returned iterator runs over the data of the node in the tree;
            if return_form==Tree.NODE, the returned iterator runs over the nodes in the tree.
        '''
        node=self.root if (node is None) else node
        queue=[(node,self[node])]
        while queue:
            if return_form==self.NODE:
                yield queue[0][0]
            elif return_form==self.DATA:
                yield queue[0][1]
            else:
                yield queue[0]
            expansion=[(i,self[i]) for i in self._children[queue[0][0]]]
            if mode is self.DEPTH:
                queue=expansion+queue[1:]
            elif mode is self.WIDTH:
                queue=queue[1:]+expansion

    def clear(self):
        '''
        Clear the contents of the tree.
        '''
        dict.clear(self)
        self.root=None
        self._parent={}
        self._children={}
