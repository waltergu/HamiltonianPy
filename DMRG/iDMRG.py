'''
Infinite DMRG.
'''

__all__=['iDMRG']

from ..Math.Tensor import *
from ..Basics import *
from MPS import *

class iDMRG(Engine):
    '''
    '''
    def __init__(self,name,lattice,term,config,**karg):
        self.name=name
        self.lattice=lattice
        self.term=term
        self.config=config
        self.table=config.table
        self.labels=[]
        self.map={}
        






        self.sys=Block()
        self.env=Block()

    def proceed(self):
        self.enlarge_lattice()
        self.find_operators()
        self.set_matrix()
        self.find_gs()
        self.truncate()

    def enlarge_lattice(self):
        sp,ep=self.lattice.point.values[0],self.lattice.point.values[0]
