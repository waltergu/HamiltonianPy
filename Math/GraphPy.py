'''
'''

class Graph(dict):
    '''
    '''

    def __init__(self):
        self.edges={}

    def __setitem__(self,node,data):
        if node in self:
            self[node]=data
        else:
            raise ValueError("Graph __setitem__ error: the node(%s) does not exist."%str(node))

    def add_node(self,node,data=None):
        super(Graph,self).__setitem__(node,data)
        self.edges[node]=[]

    def add_edge(self,origins,end):
        for origin in origins:
            if origin not in self:
                self.add_node(origin,None)
        if end not in self:
            self.add_node(origin,None)
        self.edges[end].append(origins)

    def remove_node(self,node):
        pass

    def remove_edge(self,origins,end):
        pass
