from Test.Misc import *
from Test.Basics import *
from Test.DataBase import *
from Test.TensorNetwork import *
from Test.FreeSystem import *
from Test.ED import*
from Test.VCA import *
from Test.DMRG import *
from Test.QMC import *
import sys

for arg in sys.argv:
    test_misc(arg)
    test_basics(arg)
    test_database(arg)
    test_tensornetwork(arg)
    test_fre_sys(arg)
    test_ed(arg)
    test_vca(arg)
    test_dmrg(arg)
    test_qmc(arg)
