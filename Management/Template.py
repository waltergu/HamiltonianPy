'''
Project templates, including:
    * functions: manager, basics, gitignore, license, tba, ed, vca, dmrg
'''

__all__=['manager','basics','tba','ed','vca','dmrg']

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

def basics(name):
    return  [
            "from HamiltonianPy import *",
            "",
            "__all__=['name',]",
            "",
            "# The name of the model",
            "name=%s"%name,
            "",
            "# Other basics of the model",
            "",
            ]

def tba():
    return  [
            "import numpy as np",
            "import HamiltonianPy.FreeSystem as TBA",
            "from HamiltonianPy import *",
            "from basics import *",
            "",
            "__all__=['tbaconstruct']",
            "",
            "def tbaconstruct(bc='PB',**karg):",
            "   NAME='%s_%s'%(name,bc)",
            "   lattice=Lattice(name=NAME,rcoords=,vectors=,nneighbour=)",
            "   config=IDFConfig(priority=DEFAULT_FERMIONIC_PRIORITY,pids=lattice.pids,map=)",
            "   tba=TBA.TBA(",
            "       dout=       result/tba",
            "       name=       NAME,",
            "       lattice=    lattice,",
            "       config=     config,",
            "       terms=[",
            "                   ],",
            ")",
            "   tba.register(),",
            "   tba.summary()",
            ]

def ed():
    return  [
            "import numpy as np",
            "import HamiltonianPy.ED as ED",
            "from HamiltonianPy import *",
            "from basics import *",
            "",
            "__all__=['edconstruct']",
            "",
            "def edconstruct(bc='PB',**karg):"
            "   NAME='%s_%s'%(name,bc)"
            ]

def vca():
    return  [

            ]

def dmrg():
    return  [

            ]
