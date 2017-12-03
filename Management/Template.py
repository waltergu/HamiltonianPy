'''
Project templates, including:
    * functions: manager, config, gitignore, license, tba, ed, vca, dmrg, fbfm
'''

__all__=['license','gitignore','manager','config','tba','ed','vca','dmrg','fbfm']

import datetime

license_template="""\
Copyright (C) {year} {authors}

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with this program. If not, see <http://www.gnu.org/licenses/>.
"""

gitignore_template="""\
# python
*.py[cod]
build/
data/
log/

# tex
*.aux
*.bak
*.bbl
*.out
*.sav
*.gz
*.rar
*.log
*.blg
*Notes.bib
"""

manager_template="""\
import mkl
from HamiltonianPy import *
from source import *
from collections import OrderedDict

if __name__=='__main__':
    # Forbid multithreading
    mkl.set_num_threads(1)

    # When using log files, set it to be False
    Engine.DEBUG=True

    # When using log files and data files, it's safe to set it to be True.
    Engine.MKDIR=False

    # parameters
    parameters=OrderedDict()
"""

config_template="""\
from HamiltonianPy import *

__all__=['name','nneighbour','parametermap','idfmap','qnsmap']

# The configs of the model
name=None
nneighbour=None

# parametermap
parametermap=None

# idfmap
idfmap=lambda pid: None

# qnsmap
qnsmap=lambda index: None

# terms
# example
# t=lambda **parameters: Hopping('t',parameters['t'],neighbour=1)
"""

tba_template="""\
import numpy as np
import HamiltonianPy.FreeSystem as TBA
from HamiltonianPy import *
from config import *

__all__=['tbaconstruct']

def tbaconstruct(parameters,lattice,terms,**karg):
    config=IDFConfig(priority=DEFAULT_FERMIONIC_PRIORITY,pids=lattice.pids,map=idfmap)
    tba=TBA.TBA(
        dlog=       'log',
        din=        'data',
        dout=       'result/tba',
        name=       '%s_%s'%(name,lattice.name),
        parameters= parameters,
        map=        parametermap,
        lattice=    lattice,
        config=     config,
        terms=      [term(**parameters) for term in terms],
        dtype=      np.complex128
        )
    return tba
"""

sed_template="""\
import numpy as np
import HamiltonianPy.ED as ED
from HamiltonianPy import *
from config import *

__all__=['edconstruct']

def edconstruct(parameters,lattice,sectors,terms,**karg):
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

def edconstruct(parameters,sectors,lattice,terms,**karg):
    config=IDFConfig(priority=DEFAULT_FERMIONIC_PRIORITY,pids=lattice.pids,map=idfmap)
    ed=ED.FED(
        dlog=       'log',
        din=        'data',
        dout=       'result/ed',
        name=       '%s_%s_%s'%(name,lattice.name,'_'.join(sector.rep for sector in sectors)),
        parameters= parameters,
        map=        parametermap,
        sectors=    sectors,
        lattice=    lattice,
        config=     config,
        terms=      [term(**parameters) for term in terms],
        dtype=      np.complex128
        )
    return ed
"""

vca_template="""\
import numpy as np
import HamiltonianPy.ED as ED
import HamiltonianPy.VCA as VCA
from HamiltonianPy import *
from config import *

__all__=['vcaconstruct']

def vcaconstruct(parameters,sectors,cell,lattice,terms,weiss,mask=['nambu'],**karg):
    config=IDFConfig(priority=DEFAULT_FERMIONIC_PRIORITY,pids=lattice.pids,map=idfmap)
    # edit the value of nstep if needed
    cgf=ED.FGF(operators=fspoperators(config.table(),lattice),nstep=150,method='S',prepare=ED.EDGFP,run=ED.EDGF)
    vca=VCA.VCA(
        dlog=       'log',
        din=        'data',
        dout=       'result/vca',
        cgf=        cgf,
        name=       '%s_%s_%s'%(name,lattice.name,'_'.join(sector.rep for sector in sectors)),
        parameters= parameters,
        map=        parametermap,
        sectors=    sectors,
        cell=       cell,
        lattice=    lattice,
        config=     config,
        terms=      [term(**parameters) for term in terms],
        weiss=      [term(**parameters) for term in weiss],
        mask=       mask,
        dtype=      np.complex128
        )
    return vca
"""

dmrg_template="""\
import numpy as np
import HamiltonianPy.DMRG as DMRG
from HamiltonianPy import *
from HamiltonianPy.TensorNetwork import *
from config import *

__all__=['dmrgconstruct']

def dmrgconstruct(parameters,lattice,terms,targets,core='idmrg',**karg):
    priority,layers,mask=DEGFRE_{system}_PRIORITY,DEGFRE_{system}_LAYERS,['{mask}']
    dmrg=DMRG.DMRG(
        dlog=       'log',
        din=        'data',
        dout=       'result/dmrg',
        name=       '%s_%s'%(name,lattice.name),
        parameters= parameters,
        map=        parametermap,
        mps=        MPS(mode='NB' if targets[-1] is None else 'QN'),
        lattice=    lattice,
        config=     IDFConfig(priority=priority,map=idfmap),
        degfres=    DegFreTree(mode='NB' if targets[-1] is None else 'QN',priority=priority,layers=layers,map=qnsmap),
        terms=      [term(**parameters) for term in terms],
        mask=       mask,
        dtype=      np.complex128
        )
    # edit the value of nmax and nmaxs if needed
    if core=='idmrg':
        dmrg.register(DMRG.TSG(name='GROWTH',targets=targets,nmax=100,run=DMRG.DMRGTSG))
    elif core=='fdmrg':
        dmrg.add(DMRG.TSG(name='GROWTH',targets=targets,nmax=100,plot=False,run=DMRG.DMRGTSG))
        dmrg.register(DMRG.TSS(name='SWEEP',target=targets[-1],nsite=dmrg.nspb*len(targets)*2,nmaxs=[100,100],dependences=['GROWTH'],run=DMRG.DMRGTSS))
    else:
        raise ValueError('dmrgconstruct error: not supported core %s.'%core)
    dmrg.summary()
    return dmrg
"""

fbfm_template="""\
import numpy as np
import HamiltonianPy.FBFM as FB
from HamiltonianPy import *
from config import *

__all__=['fbfmconstruct']

def fbfmconstruct(parameters,basis,lattice,terms,interactions,**karg):
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

def license(authors): return license_template.format(year=datetime.datetime.now().year,authors=authors)
def gitignore(): return gitignore_template
def manager(): return manager_template
def config(): return config_template
def tba(*arg): return tba_template
def ed(system): return sed_template if system=='spin' else fed_template
def vca(*arg): return vca_template
def dmrg(system): return dmrg_template.format(system='SPIN' if system=='spin' else 'FERMIONIC',mask='' if system=='spin' else 'nambu')
def fbfm(*arg): return fbfm_template
