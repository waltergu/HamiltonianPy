'''
fbfm template.
'''

__all__=['fbfm']

fbfm_template="""\
import numpy as np
import HamiltonianPy.FBFM as FB
from HamiltonianPy import *
from .config import *

__all__=['fbfmconstruct']

def fbfmconstruct(name,parameters,basis,lattice,terms,interactions,**karg):
    fbfm=FB.FBFM(
        dlog=           'log',
        din=            'data',
        dout=           'result/fbfm',
        name=           '%s_%s_%s'%(name,lattice.name,basis.polarization),
        parameters=     parameters,
        map=            parametermap,
        basis=          basis,
        lattice=        lattice,
        config=         IDFConfig(priority=FB.FBFM_PRIORITY,pids=lattice.pids,map=idfmap),
        terms=          [term(**parameters) for term in terms],
        interactions=   [term(**parameters) for term in interactions],
        dtype=          np.complex128
    )
    return fbfm
"""

def fbfm(**karg):
    return fbfm_template
