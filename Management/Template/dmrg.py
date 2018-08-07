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

def fdmrgconstruct(name,parameters,lattice,terms,target,maxiter=None,{argumentstatistics}boundary=None,**karg):
    priority,layers,mask=DEGFRE_{system}_PRIORITY,DEGFRE_{system}_LAYERS,[{mask}]
    config=IDFConfig(priority=priority,map=idfmap)
    degfres=DegFreTree(priority=priority,layers=layers,map=qnsmap)
    if isinstance(lattice,Cylinder):
        tsg=DMRG.TSG(name='GROWTH',target=target,maxiter=maxiter,nmax=100,plot=False,run=DMRG.fDMRGTSG)
        tss=DMRG.TSS(name='SWEEP',target=target(maxiter-1),nsite=DMRG.NS(config,degfres,lattice,mask)*2*maxiter,nmaxs=[100,100],run=DMRG.fDMRGTSS)
    else:
        tsg=None
        tss=DMRG.TSS(name='SWEEP',target=target,nsite=DMRG.NS(config,degfres,lattice,mask),nmaxs=[100,100],run=DMRG.fDMRGTSS)
    dmrg=DMRG.fDMRG(
        dlog=       'log',
        din=        'data',
        dout=       'result/fdmrg',
        name=       '%s_%s'%(name,lattice.name),
        tsg=        tsg,
        tss=        tss,
        parameters= parameters,
        map=        parametermap,
        lattice=    lattice,
        config=     config,
        degfres=    degfres,
        terms=      [term({termstatistics}**parameters) for term in terms],
        mask=       mask,
        boundary=   boundary,
        ttype=      ttype,
        dtype=      np.complex128
        )
    return dmrg
"""

idmrg_template="""\
import numpy as np
import HamiltonianPy.DMRG as DMRG
from HamiltonianPy import *
from HamiltonianPy.TensorNetwork import *
from config import *

__all__=['idmrgconstruct']

def idmrgconstruct(name,parameters,lattice,terms,target,maxiter,{argumentstatistics}boundary=None,**karg):
    priority,layers,mask=DEGFRE_{system}_PRIORITY,DEGFRE_{system}_LAYERS,[{mask}]
    dmrg=DMRG.iDMRG(
        dlog=       'log',
        din=        'data',
        dout=       'result/idmrg',
        name=       '%s_%s'%(name,lattice.name),
        tsg=        DMRG.TSG(name='ITER',target=target,maxiter=maxiter,nmax=100,plot=True,run=DMRG.iDMRGTSG),
        parameters= parameters,
        map=        parametermap,
        lattice=    lattice,
        config=     IDFConfig(priority=priority,map=idfmap),
        degfres=    DegFreTree(priority=priority,layers=layers,map=qnsmap),
        terms=      [term({termstatistics}**parameters) for term in terms],
        mask=       mask,
        boundary=   boundary,
        ttype=      ttype,
        dtype=      np.complex128
        )
    return dmrg
"""

def fdmrg(**karg):
    return fdmrg_template.format(
            argumentstatistics=     '' if karg['system']=='spin' else "statistics='f',",
            termstatistics=         '' if karg['system']=='spin' else "statistics,",
            system=                 'SPIN' if karg['system']=='spin' else 'FOCK',
            mask=                   '' if karg['system']=='spin' else "'nambu'"
            )

def idmrg(**karg):
    return idmrg_template.format(
            argumentstatistics=     '' if karg['system']=='spin' else "statistics='f',",
            termstatistics=         '' if karg['system']=='spin' else "statistics,",
            system=                 'SPIN' if karg['system']=='spin' else 'FOCK',
            mask=                   '' if karg['system']=='spin' else "'nambu'"
            )
