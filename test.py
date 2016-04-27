from Test.Math import *
from Test.Basics import *
from Test.DataBase import *
from Test.FreeSystem import *
from Test.ED import*
from Test.VCA import *
from Test.MERA import *
import sys

for arg in sys.argv:
    test_basics(arg)
    test_math(arg)
    test_database(arg)
    test_fre_sys(arg)
    test_ed(arg)
    test_vca(arg)
    test_mera(arg)
