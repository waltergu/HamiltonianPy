'''
'''

__all__=['test_vca']

def test_vca(arg):
    if arg in ('vca','all'):
        from HamiltonianPP.VCA.test.VCA import *
        test_vca_square()
#    if arg in ('vcacct','all'):
#        from HamiltonianPP.VCA.test.VCACCT import *
#        test_vcacct()
