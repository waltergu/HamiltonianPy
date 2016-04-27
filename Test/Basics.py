'''
Basics test.
'''

__all__=['test_basics']

def test_basics(arg):
    if arg in ('geometry','basics','all'):
        from HamiltonianPP.Basics.test.Geometry import *
        test_geometry()
    if arg in ('degfre','basics','all'):
        from HamiltonianPP.Basics.test.DegreeOfFreedom import *
        test_deg_fre()
    if arg in ('basespace','basics','all'):
        from HamiltonianPP.Basics.test.BaseSpace import *
        test_basespace()
    if arg in ('engineapp','basics','all'):
        from HamiltonianPP.Basics.test.EngineApp import *
        test_engineapp()
    if arg in ('operator','basics','all'):
        from HamiltonianPP.Basics.test.Operator import *
        test_operator()
    if arg in ('term','basics','all'):
        from HamiltonianPP.Basics.test.Term import *
        test_term()
    if arg in ('basisf','basics','all'):
        from HamiltonianPP.Basics.test.BasisF import *
        test_basisf()
    if arg in ('optrep','basics','all'):
        from HamiltonianPP.Basics.test.OperatorRepresentation import *
        test_opt_rep()
