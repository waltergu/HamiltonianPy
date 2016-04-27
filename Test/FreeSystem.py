'''
FreeSystem test.
'''

__all__=['test_fre_sys']

def test_fre_sys(arg):
    if arg in ('tba','fresys','all'):
        from HamiltonianPP.FreeSystem.test.TBA import *
        test_tba()
    if arg in ('flqt','fresys','all'):
        from HamiltonianPP.FreeSystem.test.FLQT import *
        test_flqt()
    if arg in ('scmf','fresys','all'):
        from HamiltonianPP.FreeSystem.test.SCMF import *
        test_scmf()
