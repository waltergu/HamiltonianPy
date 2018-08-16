if __name__=='__main__':
    import sys
    if 'clean' in sys.argv:
        import platform,os
        if platform.system()=='Windows':
            os.system("powershell.exe rm -r *.png")
            os.system("powershell.exe rm -r *.dat")
            os.system("powershell.exe rm -r *.log")
        else:
            os.system("rm -rf *.png")
            os.system("rm -rf *.dat")
            os.system("rm -rf *.log")
    else:
        from Test.test_Misc import *
        from Test.test_Basics import *
        from Test.test_TensorNetwork import *
        from Test.test_FreeSystem import *
        from Test.test_FBFM import *
        from Test.test_ED import*
        from Test.test_VCA import *
        from Test.test_DMRG import *
        from unittest import TestSuite,main
        all=TestSuite()
        all.addTest(misc)
        all.addTest(basics)
        all.addTest(tensornetwork)
        all.addTest(fresys)
        all.addTest(fbfm)
        all.addTest(ed)
        all.addTest(vcaall)
        all.addTest(dmrg)
        main(verbosity=2)
