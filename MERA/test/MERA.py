'''
MERA test.
'''

__all__=['test_mera']

from numpy import *
from HamiltonianPP.Math import Tree
from HamiltonianPP.MERA.MERAPy import *

def test_mera():
    print 'test_mera'
    nlayer,nbranch,nbranch_top=4,3,2
    N=nbranch_top*nbranch**(nlayer-1)
    points=[array([i*1.0]) for i in xrange(N)]
    A=MERA('WG',points,nlayer=nlayer,nbranch=nbranch,nbranch_top=nbranch_top)
    for tag,node in A.expand(mode=Tree.WIDTH):
        print tag,node
    A.graph()
    print
