'''
Tree structure, including:
1) classes: Node, Tree
'''

__all__=['Node','Tree']

class Node(object):
    '''
    Node class.
    Attributes:
        data: any type
            The data stored in the Node.
        parent: hashable object
            The tag of the Node's parent.
        childre: list of hashable objects
            List of the tags of the Node's children.
    '''

    def __init__(self,data,parent=None):
        '''
        Constructor.
        Parameters:
            data: any type
                The data stored in the Node.
            parent: hashable object, optional, default None
                The tag of the Node's parent.
        '''
        self.data=data
        self.parent=parent
        self.children=[]

    def __repr__(self):
        '''
        Convert an instance to string.
        '''
        return 'Node(data=%s)'%str(self.data)

    def set_parent(self,parent):
        '''
        Set the Node's parent.
        Parameters:
            parent: hashable object
                The tag of the Node's parent.
        '''
        self.parent=parent
        
    def add_child(self,child):
        '''
        Add a child of the Node.
        Parameters:
            child: hashable object
                The tag of the child to be added.
        '''
        self.children.append(child)

    def remove_child(self,child):
        '''
        Remove a child of the Node.
        Parameters:
            child: hashable object
                The tag of the child to be removed.
        '''
        self.children.remove(child)

    def update(self,data):
        '''
        Update the data stored in the Node.
        Parameters:
            data: any data
                The new data.
        '''
        self.data=data

class Tree(dict):
    '''
    Tree class.
    Attributes:
        root: hashable object
            The tag of the Tree's root node.
    '''

    (DEPTH,WIDTH)=list(range(2))
    (NID,NODE,PAIR)=list(range(3))

    def __init__(self,nid=None,node=None):
        '''
        Constructor.
        Parameters:
            nid: hashable object, optional, default None
                The tag of the Tree's root node.
            node: Node, optional, default None
                The root node.
        '''
        self.root=None
        if nid is not None:
            self[nid]=node

    def __setitem__(self,nid,node):
        '''
        Set a node by <Node>[nid]=node method.
        Parameters:
            nid: hashable object
                The tag of the node.
            node: Node
                The node.
        '''
        if isinstance(node,Node):
            if self.root is None:
                self.root=nid
                super(Tree,self).__setitem__(nid,node)
            elif node.parent in self:
                if nid not in self:
                    self[node.parent].add_child(nid)
                    super(Tree,self).__setitem__(nid,node)
                elif self[nid].parent==node.parent and self[nid].children==node.children:
                    super(Tree,self).__setitem__(nid,node)
                else:
                    raise ValueError('Tree __setitem__ error: if the node to be added has a same nid as one of those already stored in the tree, its parent and children cannot be changed.')
            else:
                raise ValueError('Tree __setitem__ error: the parent of the node does not exist.')
        else:
            raise ValueError('Tree __setitem__ error: the new item must be an instance of Node.')

    def add_subtree(self,subtree,parent=None):
        '''
        Add a subtree.
        Parameters:
            subtree: Tree
                The subtree to be added.
            parent: hashable object
                The tag of the subree's root node's parent.
        '''
        if isinstance(subtree,Tree):
            if self.root is None:
                self.root=subtree.root
                self.update(subtree)
            elif parent in self:
                if len(set(subtree.keys())&set(self.keys()))==0:
                    self[parent].add_child(subtree.root)
                    self.update(subtree)
                    self[subtree.root].set_parent(parent)
                else:
                    raise ValueError('Tree add_subtree error: the subtree to be added cannot have a node whose nid is same to one of those already stored in the tree.')
            else:
                raise ValueError('Tree add_subtree error: the parent of the node does not exist.')
        else:
            raise ValueError('Tree add_subtree error: the first parameter must be an instance of Tree.')

    def remove_subtree(self,nid):
        '''
        Remove a subtree.
        Parameters:
            nid: hashable object
                The tag of the to-be-removed subtree's root node.
        '''
        if self[nid].parent is None:
            self.root=None
            for key in self.keys():
                self.pop(key,None)
        else:
            self.parent(nid).remove_child(nid)
            for key in self.expand(nid=nid,mode=Tree.DEPTH,return_form=Tree.NID):
                self.pop(key,None)

    def parent(self,nid):
        '''
        Return the parent node of whose tag is nid.
        Parameters:
            nid: hashable object
                The tag of whose parent is inquired.
        Returns: Node
            The parent node.
        '''
        return self[self[nid].parent]

    def children(self,nid):
        '''
        Return a list of the child nodes of whose tag is nid.
        Parameters:
            nid: hashable object
                The tag of whose children are inquired.
        Returns: list of Node
            The list of child nodes.
        '''
        return [self[child] for child in self[nid].children]

    def siblings(self,nid):
        '''
        Return a list of the sibling nodes (i.e. the nodes sharing the same parent) of whose tag is nid.
        Parameters:
            nid: hashable object
                The tag of whose siblings are inquired.
        Returns: list of Node
            The list of the sibling nodes.
        '''
        if nid!=self.root:
            return [self[key] for key in self.parent(nid).children if key!=nid]
        else:
            return []

    def subtree(self,nid):
        '''
        Return a subtree whose root node's tag is nid.
        Parameters:
            nid: hashable object
                The tag of the subtree's root node.
        Returns: Tree
            The subtree.
        '''
        result=Tree()
        result.root=nid
        for key,node in self.expand(nid=nid):
            result.update({key:node})
        return result

    def expand(self,nid=None,mode=DEPTH,return_form=PAIR):
        '''
        Expand the Tree.
        Parameters:
            nid: hashable object,optional, default None
                The tag of the node with which to begin to expand the Tree.
                If it is None, the expansion begins with the root node by default.
            mode: Tree.DEPTH/Tree.WIDTH, optional, default Tree.DEPTH
                The flag to choose depth-first expansion or width-first expansion.
            return_form: Tree.PAIR/Tree.NID/Tree.NODE,optional, default Tree.PAIR
                The flag to set the form of the returned iterator.
        Returns: iterator
            if return_form==Tree.PAIR, the returned iterator runs over tuples in the form (nid,node);
            if return_form==Tree.NID, the returned iterator runs over the tags of the node in the tree;
            if return_form==Tree.NODE, the returned iterator runs over the nodes in the tree.
        '''
        nid=self.root if (nid is None) else nid
        queue=[(nid,self[nid])]
        while queue:
            if return_form==self.NID:
                yield queue[0][0]
            elif return_form==self.NODE:
                yield queue[0][1]
            else:
                yield queue[0]
            expansion=[(i,self[i]) for i in queue[0][1].children]
            if mode is self.DEPTH:
                queue=expansion+queue[1:]
            elif mode is self.WIDTH:
                queue=queue[1:]+expansion
