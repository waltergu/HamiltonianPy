import sys
for arg in sys.argv:
    if arg in ('geometry','all'):
        from Test.Geometry import *
        test_geometry()
    if arg in ('degfre','all'):
        from Test.DegreeOfFreedom import *
        test_deg_fre()
    if arg in ('basespace','all'):
        from Test.BaseSpace import *
        test_basespace()
    if arg in ('engineapp','all'):
        from Test.EngineApp import *
        test_engineapp()
    if arg in ('operator','all'):
        from Test.Operator import *
        test_operator()
    if arg in ('term','all'):
        from Test.Term import *
        test_term()
    if arg in ('tba','all'):
        from Test.TBA import *
        test_tba()

    #if arg in ('scmf','all'):
    #    from Test.SCMF import *
    #    test_scmf()
    #if arg in ('flqt','all'):
    #    from Test.FLQT import *
    #    test_flqt()


    #if arg in ('basise','all'):
    #    from Test.BasisE import *
    #    test_basise()
    #if arg in ('optrep','all'):
    #    from Test.OperatorRepresentation import *
    #    test_opt_rep()
    #if arg in ('lanczos','all'):
    #    from Test.Lanczos import *
    #    test_lanczos()
    #if arg in ('onr','all'):
    #    from Test.ONR import *
    #    test_onr()
    #if arg in ('vca','all'):
    #    from Test.VCA import *
    #    test_vca()
    #if arg in ('vcacct','all'):
    #    from Test.VCACCT import *
    #    test_vcacct()
