'''
==============
Tree structure
==============
'''

__all__=['Tree']

class Tree(dict):
    '''
    Tree class.

    Attributes
    ----------
    root : hashable object
        The Tree's root node.
    _parent_ : dict
        The lookup dict for parents.
    _children_ : dict
        The lookup dict for children.
    '''

    ROOT=None
    (DEPTH,WIDTH)=list(range(2))
    (NODE,DATA,PAIR)=list(range(3))

    def __init__(self,root=None,data=None):
        '''
        Constructor.

        Parameters
        ----------
        root : hashable object, optional
            The Tree's root node.
        data : any object, optional
            The data of the root node.
        '''
        self.root=None
        self._parent_={}
        self._children_={}
        if root is not None:
            self.addleaf(None,root,data)

    def addleaf(self,parent,leaf,data=None):
        '''
        Add a leaf for the tree.

        Parameters
        ----------
        parent : hashable object
            The parent of the new leaf.
        leaf : hashable object
            The new leaf.
        data : any object, optional
            The data of the new leaf.
        '''
        if self.root is None:
            if parent is None:
                self.root=leaf
            else:
                raise ValueError('Tree addleaf error: the parent for the first leaf of the tree must be None.')
        elif parent in self:
            self._children_[parent].append(leaf)
        else:
            raise ValueError('Tree addleaf error: the parent of the leaf does not exist.')
        self._parent_[leaf]=parent
        super(Tree,self).__setitem__(leaf,data)
        self._children_[leaf]=[]

    def addsubtree(self,subtree,parent=None):
        '''
        Add a subtree.

        Parameters
        ----------
        subtree : Tree
            The subtree to be added.
        parent : hashable object
            The parent of the subtree's root.
        '''
        if isinstance(subtree,Tree):
            if self.root is None:
                self.root=subtree.root
                self.update(subtree)
                self._parent_.update(subtree._parent_)
                self._children_.update(subtree._children_)
            elif parent in self:
                if len(set(subtree.keys())&set(self.keys()))==0:
                    self.update(subtree)
                    self._parent_.update(subtree._parent_)
                    self._children_.update(subtree._children_)
                    self._parent_[subtree.root]=parent
                    self._children_[parent].append(subtree.root)
                else:
                    raise ValueError('Tree addsubtree error: the subtree to be added cannot share any same node with the parent tree.')
            else:
                raise ValueError('Tree addsubtree error: the parent of the subtree does not exist.')
        else:
            raise ValueError('Tree addsubtree error: the first parameter must be an instance of Tree.')

    def removesubtree(self,node):
        '''
        Remove a subtree.

        Parameters
        ----------
        node : hashable object
            The root node of the to-be-removed subtree.
        '''
        if node==self.root:
            self.root=None
            for key in self.keys():
                self.pop(key,None)
                self._parent_.pop(key,None)
                self._children_.pop(key,None)
        else:
            self._children_[self._parent_[node]].remove(node)
            del self._parent_[node]
            temp=[]
            for key in self.expand(node=node,mode=Tree.DEPTH,returnform=Tree.NODE):
                temp.append(key)
            for key in temp:
                self.pop(key,None)
                self._children_.pop(key,None)

    def movesubtree(self,node,parent):
        '''
        Move a subtree to a child of a new parent.

        Parameters
        ----------
        node : hashable object
            The root node of the to-be-moved subtree.
        parent : hashable object
            The new parent of the the subtree.
        '''
        self._children_[self._parent_[node]].remove(node)
        self._parent_[node]=parent
        self._children_[parent].append(node)

    def ancestor(self,node,generation=1):
        '''
        Return the ancestor of a node.

        Parameters
        ----------
        node : hashable object
            The node whose ancestor is inquired.
        generation : positive integer, optional
            The generation of the ancestor.

        Returns
        -------
        hashable object
            The ancestor.
        '''
        assert generation>0
        result=node
        for _ in range(generation):
            result=self.parent(result)
        return result

    def parent(self,node):
        '''
        Return the parent of a node.

        Parameters
        ----------
        node : hashable object
            The node whose parent is inquired.

        Returns
        -------
        hashable object
            The parent.
        '''
        return self._parent_[node]

    def children(self,node):
        '''
        Return a list of a node's children.

        Parameters
        ----------
        node : hashable object
            The node whose children are inquired.

        Returns
        -------
        list of hashable object
            The list of the children.
        '''
        return self._children_[node]

    def descendants(self,node,generation=1):
        '''
        Return a list of a node's descendants.

        Parameters
        ----------
        node : hashable object
            The node whose descendants are inquired.
        generation : positive integer
            The generation of the descendants.

        Returns
        -------
        list of hashable object
            The descendants.
        '''
        assert generation>0
        result=[node]
        for _ in range(generation):
            result=[node for mediate in result[:] for node in self.children(mediate)]
        return result

    def siblings(self,node):
        '''
        Return a list of the siblings (i.e. the nodes sharing the same parent) of a node.

        Parameters
        ----------
        node : hashable object
            The node whose siblings are inquired.

        Returns
        -------
        list of hashable object
            The list of the siblings.
        '''
        if node!=self.root:
            return [key for key in self._children_[self._parent_[node]] if key!=node]
        else:
            return []

    def subtree(self,node):
        '''
        Return a subtree whose root is node.

        Parameters
        ----------
        node : hashable object
            The root node of the subtree.

        Returns
        -------
        Tree
            The subtree.
        '''
        result=Tree()
        result.root=node
        for node,data in self.expand(node=node,returnform=Tree.PAIR):
            super(Tree,result).__setitem__(node,data)
            result._parent_[node]=self._parent_[node]
            result._children_[node]=self._children_[node]
        return result

    def isleaf(self,node):
        '''
        Judge whether a node is a leaf (a node without children) or not.

        Parameters
        ----------
        node : any hashable object
            The node to be judged.

        Returns
        -------
        logical
            True for is False for not.
        '''
        return len(self._children_[node])==0

    @property
    def leaves(self):
        '''
        Return all the leaves contained in this tree.
        '''
        return [node for node in self if self.isleaf(node)]

    def expand(self,node=None,mode=DEPTH,returnform=PAIR):
        '''
        Expand the Tree.

        Parameters
        ----------
        node : hashable object,optional
            The node with which to begin to expand the Tree.
            If it is None, the expansion begins with the root by default.
        mode : Tree.DEPTH/Tree.WIDTH, optional
            The flag to choose depth-first expansion or width-first expansion.
        returnform: Tree.PAIR/Tree.NODE/Tree.DATA, optional
            The flag to set the form of the returned iterator.

        Yields
        ------
        iterator
            * if returnform==Tree.PAIR, the returned iterator runs over tuples in the form (node,data);
            * if returnform==Tree.DATA, the returned iterator runs over the data of the node in the tree;
            * if returnform==Tree.NODE, the returned iterator runs over the nodes in the tree.
        '''
        assert self.root is not None
        node=self.root if (node is None) else node
        queue=[(node,self[node])]
        while queue:
            if returnform==self.NODE:
                yield queue[0][0]
            elif returnform==self.DATA:
                yield queue[0][1]
            else:
                yield queue[0]
            expansion=[(i,self[i]) for i in self._children_[queue[0][0]]]
            if mode is self.DEPTH:
                queue=expansion+queue[1:]
            elif mode is self.WIDTH:
                queue=queue[1:]+expansion

    def level(self,node):
        '''
        Return the level of the node in the tree.

        Parameters
        ----------
        node : hashable object
            The node whose level is to be queried.

        Returns
        -------
        integer
            The level of the node in the tree.
        '''
        result=0
        while 1:
            node=self.parent(node)
            if node is None: break
            result+=1
        return result

    def clear(self):
        '''
        Clear the contents of the tree.
        '''
        dict.clear(self)
        self.root=None
        self._parent_={}
        self._children_={}
