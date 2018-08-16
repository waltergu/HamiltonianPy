'''
ed template.
'''

__all__=['ed']

sed_template="""\
import numpy as np
import HamiltonianPy.ED as ED
from HamiltonianPy import *
from config import *

__all__=['edconstruct']

def edconstruct(name,parameters,lattice,sectors,terms,boundary=None,**karg):
    config=IDFConfig(priority=DEFAULT_SPIN_PRIORITY,pids=lattice.pids,map=idfmap)
    qnses=None if qnsmap is None else QNSConfig(indices=config.table().keys(),priority=DEFAULT_SPIN_PRIORITY,map=qnsmap)
    ed=ED.SED(
        dlog=       'log',
        din=        'data',
        dout=       'result/ed',
        name=       '%s_%s_%s'%(name,lattice.name,'_'.join(str(sector) for sector in sectors)),
        parameters= parameters,
        map=        parametermap,
        lattice=    lattice,
        config=     config,
        qnses=      qnses,
        sectors=    sectors,
        terms=      [term(**parameters) for term in terms],
        boundary=   boundary,
        dtype=      np.complex128
        )
    return ed
"""

fed_template="""\
import numpy as np
import HamiltonianPy.ED as ED
from HamiltonianPy import *
from config import *

__all__=['edconstruct']

def edconstruct(name,parameters,sectors,lattice,terms,statistics,boundary=None,**karg):
    config=IDFConfig(priority=DEFAULT_FOCK_PRIORITY,pids=lattice.pids,map=idfmap)
    ed=ED.FED(
        dlog=       'log',
        din=        'data',
        dout=       'result/ed',
        name=       '%s_%s_%s'%(name,lattice.name,'_'.join(repr(sector) for sector in sectors)),
        parameters= parameters,
        map=        parametermap,
        sectors=    sectors,
        lattice=    lattice,
        config=     config,
        terms=      [term(statistics,**parameters) for term in terms],
        boundary=   boundary,
        dtype=      np.complex128
        )
    return ed
"""

def ed(**karg):
    return sed_template if karg['system']=='spin' else fed_template
