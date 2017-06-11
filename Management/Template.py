'''
Project templates, including:
    * functions: manager, config, gitignore, license, tba, ed, vca, dmrg
'''

__all__=['license','gitignore','manager','config','tba','ed','vca','dmrg']

import datetime

def license(authors):
    return  [
            "Copyright (C) %s %s"%(datetime.datetime.now().year,authors),
            "",
            "This program is free software: you can redistribute it and/or modify",
            "it under the terms of the GNU General Public License as published by",
            "the Free Software Foundation, either version 3 of the License, or",
            "(at your option) any later version.",
            "",
            "This program is distributed in the hope that it will be useful,",
            "but WITHOUT ANY WARRANTY; without even the implied warranty of",
            "MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the",
            "GNU General Public License for more details.",
            "",
            "You should have received a copy of the GNU General Public License",
            "along with this program.  If not, see <http://www.gnu.org/licenses/>.",
            ]

def gitignore():
    return  ["# python","*.py[cod]","build/","data/","log/","","# tex","*.aux","*.bak","*.bbl","*.out","*.sav","*.gz","*.rar","*.log","*.blg"]

def manager():
    return  [
            "from HamiltonianPy import *",
            "from HamiltonianPy.Misc import mpirun",
            "from source import *",
            "import numpy as np",
            "import mkl",
            "",
            "if __name__=='__main__':",
            "    mkl.set_num_threads(1)",
            "",
            "    # When using log files, set it to be False",
            "    Engine.DEBUG=True",
            "",
            "    # Run the engines. Replace 'f' with the correct function",
            "    #mpirn(f,parameters,bcast=True)",
            ]

def config():
    return  [
            "from HamiltonianPy import *",
            "",
            "__all__=['name','nneighbour','idfmap','qnsmap']",
            "",
            "# The configs of the model",
            "name=",
            "nneighbour=",
            "",
            "# idfmap",
            "idfmap=lambda pid: None",
            "",
            "# qnsmap",
            "qnsmap=lambda index: None",
            "",
            "# terms",
            "",
            ]

def tba(system):
    return  [
            "import numpy as np",
            "import HamiltonianPy.FreeSystem as TBA",
            "from HamiltonianPy import *",
            "from config import *",
            "",
            "__all__=['tbaconstruct']",
            "",
            "def tbaconstruct(parameters,lattice,terms,**karg):",
            "    config=IDFConfig(priority=DEFAULT_FERMIONIC_PRIORITY,pids=lattice.pids,map=idfmap)",
            "    tba=TBA.TBA(",
            "        dlog=       'log/tba',",
            "        din=        'data/tba',",
            "        dout=       'result/tba',",
            "        log=        '%s_%s_%s_TBA.log'%(name,lattice.name,parameters),",
            "        name=       '%s_%s'%(name,lattice.name),",
            "        lattice=    lattice,",
            "        config=     config,",
            "        terms=      [term(*parameters) for term in terms],",
            "        dtype=      np.complex128",
            "        )",
            "    return tba",
            ]

def ed(system):
    return  [
            "import numpy as np",
            "import HamiltonianPy.ED as ED",
            "from HamiltonianPy import *",
            "from config import *",
            "",
            "__all__=['edconstruct']",
            "",
            "def edconstruct(parameters,%s,%s,terms,**karg):"%(('lattice','target') if system=='spin' else ('basis','lattice')),
            "    config=IDFConfig(priority=DEFAULT_%s_PRIORITY,pids=lattice.pids,map=idfmap)"%('SPIN' if system=='spin' else 'FERMIONIC'),
            "    qnses=QNSConfig(indices=config.table().keys(),priority=DEFAULT_SPIN_PRIORITY,map=qnsmap)" if system=='spin' else None,
            "    ed=ED.%s("%('SED' if system=='spin' else 'FED'),
            "        dlog=       'log/ed',",
            "        din=        'data/ed',",
            "        dout=       'result/ed',",
            "        log=        %s%s%s"%("'%s_%s_%s_%s_ED.log'%(name,lattice.name,",'repr(target)' if system=='spin' else 'basis.rep',',parameters),'),
            "        name=       %s%s%s"%("'%s_%s_%s'%(name,lattice.name,",'repr(target)' if system=='spin' else 'basis.rep','),'),
            "        %s"%('qnses=      qnses,' if system=='spin' else 'basis=      basis,'),
            "        lattice=    lattice,",
            "        config=     config,",
            "        target=     target," if system=='spin' else None,
            "        terms=      [term(*parameters) for term in terms],",
            "        dtype=      np.complex128",
            "        )",
            "    return ed",
            ]

def vca(system):
    return  [
            "import numpy as np",
            "import HamiltonianPy.ED as ED",
            "import HamiltonianPy.VCA as VCA",
            "from HamiltonianPy import *",
            "from config import *",
            "",
            "__all__=['vcaconstruct']",
            "",
            "def vcaconstruct(parameters,basis,cell,lattice,terms,weiss,mask=['nambu'],**karg):",
            "    config=IDFConfig(priority=DEFAULT_FERMIONIC_PRIORITY,pids=lattice.pids,map=idfmap)",
            "    # edit the value of nstep if needed",
            "    cgf=ED.FGF(operators=fspoperators(config.table(),lattice),nstep=150,prepare=ED.EDGFP,run=ED.EDGF)",
            "    vca=VCA.VCA(",
            "        dlog=       'log/vca',",
            "        din=        'data/vca',",
            "        dout=       'result/vca',",
            "        log=        '%s_%s_%s_%s_VCA.log'%(name,lattice.name,basis.rep,parameters),",
            "        cgf=        cgf,",
            "        name=       '%s_%s_%s'%(name,lattice.name,basis.rep),",
            "        basis=      basis,",
            "        cell=       cell,",
            "        lattice=    lattice,",
            "        config=     config,",
            "        terms=      [term(*parameters) for term in terms],",
            "        weiss=      [term(*parameters) for term in weiss],",
            "        mask=       mask,",
            "        dtype=      np.complex128",
            "        )",
            "    return vca"
            ]

def dmrg(system):
    return  [
            "import numpy as np",
            "import HamiltonianPy.DMRG as DMRG",
            "from HamiltonianPy import *",
            "from HamiltonianPy.TensorNetwork import *",
            "from config import *",
            "",
            "__all__=['dmrgconstruct']",
            "",
            "def dmrgconstruct(parameters,lattice,terms,targets,core='idmrg',**karg):",
            "    priority,layers,mask=DEGFRE_%s_PRIORITY,DEGFRE_%s_LAYERS,[%s]"%(('SPIN','SPIN','') if system=='spin' else ('FERMIONIC','FERMIONIC',"'nambu'")),
            "    dmrg=DMRG.DMRG(",
            "        dlog=       'log/dmrg',",
            "        din=        'data/dmrg',",
            "        dout=       'result/dmrg',",
            "        log=        '%s_%s_%s_%s_DMRG.log'%(name,lattice.name.replace('+',str(2*len(targets))),parameters,repr(targets[-1])),",
            "        name=       '%s_%s'%(name,lattice.name),",
            "        mps=        MPS(mode='NB' if targets[-1] is None else 'QN'),",
            "        lattice=    lattice,",
            "        config=     IDFConfig(priority=priority,map=idfmap),",
            "        degfres=    DegFreTree(mode='NB' if targets[-1] is None else 'QN',priority=priority,layers=layers,map=qnsmap),",
            "        terms=      [term(*parameters) for term in terms],",
            "        mask=       mask,",
            "        dtype=      np.complex128",
            "        )",
            "    # edit the value of nmax and nmaxs if needed",
            "    if core=='idmrg':",
            "        tsg=DMRG.TSG(name='GROWTH',targets=targets,nmax=100,run=DMRG.DMRGTSG)",
            "        dmrg.register(tsg)",
            "    elif core=='fdmrg':",
            "        tsg=DMRG.TSG(name='GROWTH',targets=targets,nmax=100,plot=False,run=DMRG.DMRGTSG)",
            "        tss=DMRG.TSS(name='SWEEP',target=targets[-1],nsite=dmrg.nspb*len(targets)*2,nmaxs=[100,100],dependences=[tsg],run=DMRG.DMRGTSS)",
            "        dmrg.register(tss)",
            "    else:",
            "        raise ValueError('dmrgconstruct error: not supported core %s.'%core)",
            "    dmrg.summary()",
            "    return dmrg"
            ]
