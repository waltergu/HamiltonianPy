'''
FBasis test.
'''
__all__=['test_fbasis']

from HamiltonianPy.Basics.FermionicPackage.Basis import *

def test_fbasis():
    print 'test_fbasis'
    m,n=2,1
    bases=[     FBasis('FS',nstate=m*2,nparticle=n*2,spinz=1),
                FBasis('FP',nstate=m*2,nparticle=n*2),
                FBasis('FG',nstate=m*2),
                FBasis('FGS',nstate=m*2),
                FBasis('FGP',nstate=m*2)
                ]
    for basis in bases:
        for bs in basis.iter():
            print bs.rep
            print bs
            print
    print
