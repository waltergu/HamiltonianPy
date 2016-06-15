'''
Block test.
'''

__all__=['test_block']

from numpy import *
from HamiltonianPy.Basics import *
from HamiltonianPy.DMRG import *

def test_block():
    print 'test_block'
    qn1=QuantumNumber([('Sz',1,'U1')])
    qn2=QuantumNumber([('Sz',0,'U1')])
    qn3=QuantumNumber([('Sz',-1,'U1')])
    a=QuantumNumberCollection([(qn1,1),(qn2,1),(qn3,1)])
    print 'a:%s'%a
    print 'a+a:%s'%(a+a)

    sz=SpinMatrix((1,'z'),dtype=float64)
    sp=SpinMatrix((1,'+'),dtype=float64)
    sm=SpinMatrix((1,'-'),dtype=float64)
    print 'sz:\n%s'%sz
    print 'sp:\n%s'%sp
    print 'sm:\n%s'%sm
    matrix=kron(sz,sz,a,a,a+a,target=qn2,format='csr')+kron(sp,sm,a,a,a+a,target=qn2,format='csr')+kron(sm,sp,a,a,a+a,target=qn2,format='csr')
    print matrix.todense()
    print
