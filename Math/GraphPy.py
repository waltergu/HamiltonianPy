'''
'''

class Node(object):
    '''
    '''
    def __init__(self,data):
        

class Graph(object):
    '''
    '''
    
    def __init__(self):
        self.nodes={}
        self.edges={}

    def add_node(self,node):
        if node not in self.nodes:
            self.nodes[node]=None

    def add_edge(self,edge):
        node1,node2=edge
        if node1 in self.nodes:
            
