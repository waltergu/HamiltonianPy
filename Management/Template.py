'''
Project templates, including:
    * functions: manager, config, gitignore, license, tba, ed, vca, dmrg
'''

__all__=['manager','config','tba','ed','vca','dmrg']

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
            "# Parameters of the model",
            "#parameters=[]",
            "",
            "# Run the engines. Replace 'f' with the correct function",
            "#mpirn(f,parameters,bcast=True)",
            ]

def config():
    return  [
            "from HamiltonianPy import *",
            "",
            "__all__=['name','nneighbour']",
            "",
            "# The configs of the model",
            "name=",
            "nneighbour=",
            ]

def tba():
    return  [
            "import numpy as np",
            "import HamiltonianPy.FreeSystem as TBA",
            "from HamiltonianPy import *",
            "from config import *",
            "",
            "__all__=['tbaconstruct']",
            "",
            "def tbaconstruct(parameters,lattice,**karg):",
            "    config=IDFConfig(priority=DEFAULT_FERMIONIC_PRIORITY,pids=lattice.pids,map=)",
            "    tba=TBA.TBA(",
            "        dout=       result/tba",
            "        name=       '%s_%s'%(name,lattice.name),",
            "        lattice=    lattice,",
            "        config=     config,",
            "        terms=[",
            "                    ],",
            "        )",
            "    tba.register()",
            "    tba.summary()",
            ]

def ed():
    return  [
            "import numpy as np",
            "import HamiltonianPy.ED as ED",
            "from HamiltonianPy import *",
            "from config import *",
            "",
            "__all__=['edconstruct']",
            "",
            "def edconstruct(parameters,basis,lattice,**karg):",
            "    config=IDFConfig(priority=DEFAULT_FERMIONIC_PRIORITY,pids=lattice.pids,map=)",
            "    ed=ED.ED(",
            "        dout=       result/ed,",
            "        din=        result/ed/coeff,",
            "        name=       '%s_%s_%s'%(name,lattice.name,basis.rep),",
            "        basis=      basis,",
            "        lattice=    lattice,",
            "        config=     config,",
            "        terms=[",
            "                    ],",
            "        )",
            "    ed.register()",
            "    ed.summary()",
            ]

def vca():
    return  [
            "import numpy as np",
            "import HamiltonianPy.ED as ED",
            "import HamiltonianPy.VCA as VCA",
            "from HamiltonianPy import *",
            "from config import *",
            "",
            "__all__=['vcaconstruct']",
            "",
            "def vcaconstruct(parameters,basis,cell,lattice,**karg):",
            "    config=IDFConfig(priority=DEFAULT_FERMIONIC_PRIORITY,pids=lattice.pids,map=)",
            "    cgf=ED.GF(operators=fspoperators(config.table(),lattice),nspin=,mask=,nstep=,save_data=,prepare=ED.EDGFP,run=ED.EDGF)",
            "    vca=VCA.VCA(",
            "        dout=       result/vca,",
            "        din=        result/vca/coeff,",
            "        cgf=        cgf,",
            "        name=       '%s_%s_%s'%(name,lattice.name,basis.rep),",
            "        basis=      basis",
            "        cell=       cell,",
            "        lattice=    lattice,",
            "        config=     config,",
            "        terms=[",
            "                    ],",
            "        weiss=[",
            "                    ],",
            "        )",
            "    vca.register()",
            "    vca.summary()",
            ]

def dmrg():
    return  [
            "import numpy as np",
            "import HamiltonianPy.DMRG as DMRG",
            "from HamiltonianPy import *",
            "from HamiltonianPy.TensorNetwork import *",
            "from config import *",
            "",
            "__all__=['dmrgconstruct']",
            "",
            "def dmrgconstruct(parameters,lattice,nb,target):",
            "    priority,layers=",
            "    dmrg=DMRG.DMRG(",
            "        log=       Log('%s-%s-%s-%s.log'%(name,lattice.name.replace('+',str(nb)),'_'.join(str(p) for p in parameters),None if target is None else tuple(target))),",
            "        din=       data/dmrg,",
            "        dout=      data/dmrg,",
            "        name=      '%s-%s'%(name,lattice.name),",
            "        mps=       MPS(mode='NB' if target is None else 'QN'),",
            "        lattice=   lattice,",
            "        config=    IDFConfig(priority=priority,map=),",
            "        degfres=   DegFreTree(mode='NB' if target is None else 'QN',priority=priority,layers=layers,map=),",
            "        terms=[",
            "                    ],",
            "        mask=      [],",
            "        dtype=     np.complex128",
            "        )",
            "    targets=[]",
            "    tsg=DMRG.TSG(name='GROWTH',targets=targets,nspb=,nmax=,save_data=,plot=,save_fig=,run=DMRG.DMRGTSG)",
            "    tss=DMRG.TSS(name='SWEEP',target=targets[-1],nsite=,nmaxs=,save_data=,plot=,save_fig=,dependences=[tsg],run=DMRGTSS)",
            "    dmrg.register()",
            "    dmrg.summary()",
            ]
