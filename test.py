from Test.Misc import *
from Test.Basics import *
from Test.TensorNetwork import *
from Test.FreeSystem import *
from Test.FBFM import *
from Test.ED import*
from Test.VCA import *
from Test.DMRG import *
import sys

for arg in sys.argv:
    test_misc(arg)
    test_basics(arg)
    test_tensornetwork(arg)
    test_fre_sys(arg)
    test_fbfm(arg)
    test_ed(arg)
    test_vca(arg)
    test_dmrg(arg)
