'''
Spin wave theory for spin reserved flat band ferromagnets.
    * constants: FBFM_PRIORITY
    * classes: 
    * functions: 
'''

__all__=[]

import numpy as np
import HamiltonianPy as HP
import HamiltonianPy.FreeSystem as TBA

FBFM_PRIORITY=('spin','scope','nambu','site','orbital')

class FBFM(TBA.TBA):
    def __init__(self,lattice=None,config=None,terms=None,interactions=None):
        
        super(FBFM,self).__init__(lattice=lattice,config=config,terms=terms)
        self.igenerator=Generator
        self.generator=Generator(bonds=lattice.bonds,config=config,table=config.table(mask=mask),terms=terms)
        self.status.update(const=self.generator.parameters['const'],alter=self.generator.parameters['alter'])
    
