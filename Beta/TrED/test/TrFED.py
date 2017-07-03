'''
TrFED test.
'''

__all__=['test_trfed']

import numpy as np
from HamiltonianPy import *
from HamiltonianPy.ED import *
from HamiltonianPy.Beta.TrED import *
from scipy.linalg import eigh,norm
import time

def test_trfed():
    print 'test_trfed'
    test_fbasis()
    test_fed()

def test_fbasis():
    print 'test_fbasis'
    m=16
    t1=time.time()
    nmbasis=FBasis(up=(m,m/4),down=(m,m/4))
    t2=time.time()
    print 'NRM basis: %ss.'%(t2-t1)
    t3=time.time()
    trbasis=TRBasis(nmbasis,dk=4,nk=8)
    t4=time.time()
    print 'TRI basis: %ss'%(t4-t3)
    print

def test_fed():
    print 'test_fed'
    t1,U,m=1.0,4.0,4
    lattice=Square('S1')('%sP-1O'%m)
    config=IDFConfig(priority=DEFAULT_FERMIONIC_PRIORITY,pids=lattice.pids,map=lambda pid: Fermi(norbital=1,nspin=2,nnambu=1))

    basis=FBasis(up=(m,m/2),down=(m,m/2))
    ed=FED(name='OneD_%s_%s'%(lattice.name,basis.rep),basis=basis,lattice=lattice,config=config,terms=[Hopping('t1',t1),Hubbard('U',U)],dtype=np.complex128)
    ed.set_matrix()
    eigvals0=eigh(ed.matrix.todense(),eigvals_only=True)

    basis=TRBasis(FBasis(up=(m,m/2),down=(m,m/2)),dk=2,nk=m)
    ed=TrFED(name='OneD_%s_%s'%(lattice.name,basis.rep),basis=basis,lattice=lattice,config=config,terms=[Hopping('t1',t1),Hubbard('U',U)],dtype=np.complex128)
    eigvals1=[]
    for k in xrange(m):
        ed.set_matrix(k)
        eigvals1.append(eigh(ed.matrix.todense(),eigvals_only=True))
    eigvals1=sorted(np.concatenate(eigvals1))
    print 'diff: %s'%norm(eigvals0-eigvals1)
    print

if __name__=='__main__':
    test_trfed()
