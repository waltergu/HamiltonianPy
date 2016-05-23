'''
Multi-scale entanglement renormalization ansatz, including:
1) MERA
'''

__all__=['MERA']

from numpy import *
from ..Math.Tree import *
import matplotlib.pyplot as plt 

class MERA(Tree):
    '''
    '''
    def __init__(self,name,points,nlayer=2,nbranch=2,nbranch_top=2):
        '''
        '''
        if len(points)!=nbranch_top*nbranch**(nlayer-1):
            raise ValueError('MERA init error: the number of points(%s) does not equal the number of nodes(%s) in the zero-th layer.'%(len(points),nbranch_top*nbranch**(nlayer-1)))
        self.root=None
        self.name=name
        self.nlayer=nlayer
        self.nbranch=nbranch
        self.nbranch_top=nbranch_top
        self.generate_tree(points)
        self.generate_isometries()

    def generate_tree(self,points):
        for layer in xrange(self.nlayer,-1,-1):
            if layer==self.nlayer:
                nbranch_current=self.nbranch_top
                queue=[((layer,0),Node(data=None,parent=None))]
            else:
                nbranch_current=self.nbranch
                queue=buff
            flag=True if layer>0 else False
            count,buff=0,[]
            for tag,node in queue:
                self[tag]=node
                if flag:
                    for branch in xrange(nbranch_current):
                        buff.append(((layer-1,count),Node(data=None,parent=tag)))
                        count=count+1
        for layer,nodes in enumerate(self.layered_list):
            if layer==0:
                for node,point in zip(nodes,points):
                    node.update(point)
            else:
                for count,node in enumerate(nodes):
                    point=zeros(len(points[0]))
                    children=self[(layer,count)].children
                    for child in children:
                        point+=self[child].data
                    node.update(point/len(children))

    def generate_isometries(self):
        self.isometries={}

    def graph(self):
        if len(self[self.root].data)!=1:
            raise ValueError("MERA graph error: only 1D MERA graph is supported.")
        for tag,node in self.expand(mode=Tree.DEPTH):
            plt.scatter(node.data[0],tag[0]*1.0,zorder=3)
            for child in node.children:
                plt.plot([node.data[0],self[child].data[0]],[tag[0]*1.0,child[0]*1.0],zorder=2)
        plt.show()

    @property
    def layered_list(self):
        result=[[] for i in xrange(self.nlayer+1)]
        for tag,node in sorted(self.items(),key=lambda pair: pair[0]):
            result[tag[0]].append(node)
        return result

    @property
    def layered_dict(self):
        result={}
        for tag,node in self.iteritem():
            result[tag[0]][tag[1]]=node
        return result
