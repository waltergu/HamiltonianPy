'''
dmrg template.
'''

__all__=['fdmrg','idmrg']

fdmrg_template="""\
import numpy as np
import HamiltonianPy.DMRG as DMRG
from HamiltonianPy import *
from HamiltonianPy.TensorNetwork import *
from config import *

__all__=['fdmrgconstruct']

def fdmrgconstruct(name,parameters,lattice,terms,target,maxiter,{statistics}boundary=None,**karg):
    priority,layers,mask=DEGFRE_{system}_PRIORITY,DEGFRE_{system}_LAYERS,[{mask}]
    dmrg=DMRG.fDMRG(
        dlog=       'log',
        din=        'data',
        dout=       'result/fdmrg',
        name=       '%s_%s'%(name,lattice.name),
        parameters= parameters,
        map=        parametermap,
        lattice=    lattice,
        config=     IDFConfig(priority=priority,map=idfmap),
        degfres=    DegFreTree(mode='QN' if qnon else 'NB',priority=priority,layers=layers,map=qnsmap),
        terms=      [term({statistics}**parameters) for term in terms],
        mask=       mask,
        boundary=   boundary,
        dtype=      np.complex128
        )
    dmrg.add(DMRG.TSG(name='GROWTH',target=target,maxiter=maxiter,nmax=100,plot=False,run=DMRG.fDMRGTSG))
    dmrg.add(DMRG.TSS(name='SWEEP',target=target(maxiter-1),nsite=dmrg.nspb*maxiter*2,nmaxs=[100,100],dependences=['GROWTH'],run=DMRG.fDMRGTSS))
    return dmrg
"""

idmrg_template="""\
import numpy as np
import HamiltonianPy.DMRG as DMRG
from HamiltonianPy import *
from HamiltonianPy.TensorNetwork import *
from config import *

__all__=['idmrgconstruct']

def idmrgconstruct(name,parameters,lattice,terms,target,maxiter,{statistics}boundary=None,**karg):
    priority,layers,mask=DEGFRE_{system}_PRIORITY,DEGFRE_{system}_LAYERS,[{mask}]
    dmrg=DMRG.iDMRG(
        dlog=       'log',
        din=        'data',
        dout=       'result/idmrg',
        name=       '%s_%s'%(name,lattice.name),
        parameters= parameters,
        map=        parametermap,
        lattice=    lattice,
        config=     IDFConfig(priority=priority,map=idfmap),
        degfres=    DegFreTree(mode='QN' if qnon else 'NB',priority=priority,layers=layers,map=qnsmap),
        terms=      [term({statistics}**parameters) for term in terms],
        mask=       mask,
        boundary=   boundary,
        dtype=      np.complex128
        )
    dmrg.add(DMRG.TSG(name='ITER',target=target,maxiter=maxiter,nmax=100,plot=True,run=DMRG.iDMRGTSG))
    return dmrg
"""

def fdmrg(**karg):
    return fdmrg_template.format(
            statistics= '' if karg['system']=='spin' else 'statistics,',
            system=     'SPIN' if karg['system']=='spin' else 'FOCK',
            mask=       '' if karg['system']=='spin' else "'nambu'"
            )

def idmrg(**karg):
    return idmrg_template.format(
            statistics= '' if karg['system']=='spin' else 'statistics,',
            system=     'SPIN' if karg['system']=='spin' else 'FOCK',
            mask=       '' if karg['system']=='spin' else "'nambu'"
            )
