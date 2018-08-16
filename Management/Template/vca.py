'''
vca template.
'''

__all__=['vca','vcacct']

vca_template="""\
import numpy as np
import HamiltonianPy.ED as ED
import HamiltonianPy.VCA as VCA
from HamiltonianPy import *
from config import *

__all__=['vcaconstruct']

def vcaconstruct(name,parameters,sectors,cell,lattice,terms,weiss,baths=(),mask=('nambu',),**karg):
    config=IDFConfig(priority=DEFAULT_FOCK_PRIORITY,pids=lattice.pids,map=idfmap)
    vca=VCA.VCA(
        dlog=       'log',
        din=        'data',
        dout=       'result/vca',
        name=       '%s_%s_%s'%(name,lattice.name,'_'.join(repr(sector) for sector in sectors)),
        cgf=        VCA.VGF(nstep=150,method='S',prepare=ED.EDGFP,run=ED.EDGF),
        parameters= parameters,
        map=        parametermap,
        sectors=    sectors,
        cell=       cell,
        lattice=    lattice,
        config=     config,
        terms=      [term(**parameters) for term in terms],
        weiss=      [term(**parameters) for term in weiss],
        baths=      [term(**parameters) for term in baths],
        mask=       mask,
        dtype=      np.complex128
        )
    return vca
"""

vcacct_template="""\
import numpy as np
import HamiltonianPy.ED as ED
import HamiltonianPy.VCA as VCA
from HamiltonianPy import *
from config import *

__all__=['vcacctconstruct']

def vcacctconstruct(name,parameters,cell,lattice,terms,weiss,baths=(),subsystems=(),mask=('nambu',),**karg):
    config=IDFConfig(priority=DEFAULT_FOCK_PRIORITY,pids=lattice.pids,map=idfmap)
    vcacct=VCA.VCACCT(
        dlog=       'log',
        din=        'data',
        dout=       'result/vcacct',
        name=       '%s_%s'%(name,lattice.name),
        cgf=        VCA.VGF(nstep=150,method='S',prepare=VCA.VCACCTGFP,run=VCA.VCACCTGF),
        parameters= parameters,
        map=        parametermap,
        cell=       cell,
        lattice=    lattice,
        config=     config,
        terms=      [term(**parameters) for term in terms],
        weiss=      [term(**parameters) for term in weiss],
        baths=      [term(**parameters) for term in baths],
        subsystems= subsystems,
        mask=       mask,
        dtype=      np.complex128
        )
    return vcacct
"""

def vca(**karg):
    return vca_template

def vcacct(**karg):
    return vcacct_template
