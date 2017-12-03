'''
FBasis test.
'''
__all__=['test_fbasis']

from HamiltonianPy.Basics.FermionicPackage.Basis import *

def test_fbasis():
    print 'test_fbasis'
    m,n=2,1
    for basis in [FBasis(nstate=m*2,nparticle=n*2,spinz=1),FBasis(nstate=m*2,nparticle=n*2),FBasis(nstate=m*2)]:
        print '%s\n%s\n'%(basis.rep,basis)
    for basis in FBases(mode='FS',nstate=m*2,select=lambda n,sz: True if n%2==0 and sz==0 else False):
        print '%s\n%s\n'%(basis.rep,basis)
    for basis in FBases(mode='FP',nstate=m*2):
        print '%s\n%s\n'%(basis.rep,basis)
    for basis in FBases(mode='FG',nstate=m*2):
        print '%s\n%s\n'%(basis.rep,basis)
    print
