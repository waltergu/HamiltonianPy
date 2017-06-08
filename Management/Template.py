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
            "mkl.set_num_threads(1)",
            "",
            "# When using log files, set it to be False",
            "Engine.DEBUG=True",
            "",
            "# Run the engines. Replace 'f' with the correct function",
            "#mpirn(f,parameters,bcast=True)",
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
            "    assert len(parameters)==len(terms)",
            "    config=IDFConfig(priority=DEFAULT_FERMIONIC_PRIORITY,pids=lattice.pids,map=idfmap)",
            "    tba=TBA.TBA(",
            "        dout=       'result/tba',",
            "        name=       '%s_%s'%(name,lattice.name),",
            "        lattice=    lattice,",
            "        config=     config,",
            "        terms=      [term(parameter) for term,parameter in zip(terms,parameters)],",
            "        dtype=      np.complex128",
            "        )",
            "    # edit tasks",
            "    tba.register()",
            "    tba.summary()",
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
            "    assert len(parameters)==len(terms)",
            "    config=IDFConfig(priority=DEFAULT_%s_PRIORITY,pids=lattice.pids,map=idfmap)"%('SPIN' if system=='spin' else 'FERMIONIC'),
            "    qnses=QNSConfig(indices=config.table().keys(),priority='DEFAULT_SPIN_PRIORITY',map=qnsmap)" if system=='spin' else None,
            "    ed=ED.%s("%('SED' if system=='spin' else 'FED'),
            "        dout=       'result/ed',",
            "        din=        'data/ed',",
            "        name=       %s%s%s"%("'%s_%s_%s'%(name,lattice.name,",'repr(target)' if system=='spin' else 'basis.rep','),'),
            "        %s"%('qnses=      qnses,' if system=='spin' else 'basis=      basis,'),
            "        lattice=    lattice,",
            "        config=     config,",
            "        target=     target," if system=='spin' else None,
            "        terms=      [term(parameter) for term,parameter in zip(terms,parameters)],",
            "        dtype=      np.complex128",
            "        )",
            "    # edit tasks",
            "    ed.register()",
            "    ed.summary()",
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
            "    assert len(parameters)==len(terms)+len(weiss)",
            "    config=IDFConfig(priority=DEFAULT_FERMIONIC_PRIORITY,pids=lattice.pids,map=idfmap)",
            "    cgf=ED.FGF(operators=fspoperators(config.table(),lattice),nstep=150,prepare=ED.EDGFP,run=ED.EDGF)",
            "    vca=VCA.VCA(",
            "        dout=       'result/vca',",
            "        din=        'data/vca',",
            "        cgf=        cgf,",
            "        name=       '%s_%s_%s'%(name,lattice.name,basis.rep),",
            "        basis=      basis,",
            "        cell=       cell,",
            "        lattice=    lattice,",
            "        config=     config,",
            "        terms=      [term(parameter) for term,parameter in zip(terms,parameters[:len(terms)])],",
            "        weiss=      [term(parameter) for term,parameter in zip(weiss,parameters[len(terms):])],",
            "        mask=       mask,",
            "        dtype=      np.complex128",
            "        )",
            "    # edit tasks",
            "    vca.register()",
            "    vca.summary()",
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
            "    assert len(parameters)==len(terms)",
            "    priority,layers,mask=DEGFRE_%s_PRIORITY,DEGFRE_%s_LAYERS,[%s]"%(('SPIN','SPIN','') if system=='spin' else ('FERMIONIC','FERMIONIC',"'nambu'")),
            "    dmrg=DMRG.DMRG(",
            "        log=        Log('log/%s-%s-%s-%s.log'%(name,lattice.name.replace('+',str(2*len(targets))),parameters,repr(targets[-1]))),",
            "        din=        'data/dmrg',",
            "        dout=       'result/dmrg',",
            "        name=       '%s_%s'%(name,lattice.name),",
            "        mps=        MPS(mode='NB' if targets[-1] is None else 'QN'),",
            "        lattice=    lattice,",
            "        config=     IDFConfig(priority=priority,map=idfmap),",
            "        degfres=    DegFreTree(mode='NB' if targets[-1] is None else 'QN',priority=priority,layers=layers,map=qnsmap),",
            "        terms=      [term(parameter) for term,parameter in zip(terms,parameters)],",
            "        mask=       mask,",
            "        dtype=      np.complex128",
            "        )",
            "    if core=='idmrg':",
            "        # edit 'nspb' and 'nmax'",
            "        tsg=DMRG.TSG(name='GROWTH',targets=targets,nspb=,nmax=,run=DMRG.DMRGTSG)",
            "        dmrg.register(tsg)",
            "    elif core=='fdmrg':",
            "        # edit 'nspb', 'nmax', 'nsite' and 'nmaxs'",
            "        tsg=DMRG.TSG(name='GROWTH',targets=targets,nspb=,nmax=,plot=False,run=DMRG.DMRGTSG)",
            "        tss=DMRG.TSS(name='SWEEP',target=targets[-1],nsite=,nmaxs=,dependences=[tsg],run=DMRGTSS)",
            "        dmrg.register(tss)",
            "    else:",
            "        raise ValueError('dmrgconstruct error: not supported core %s.'%core)",
            "    dmrg.summary()",
            ]
