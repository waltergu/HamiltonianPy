'''
FOperator test.
'''

__all__=['test_foperator']

from HamiltonianPy.Basics.Geometry import *
from HamiltonianPy.Basics.DegreeOfFreedom import *
from HamiltonianPy.Basics.Operator import *
from HamiltonianPy.Basics.FermionicPackage import *

def test_foperator():
    print 'test_foperator'
    a=FQuadratic(
        value=      1.0j,
        indices=    [   Index(PID(site=1),FID(orbital=0,spin=0,nambu=CREATION)),
                        Index(PID(site=1),FID(orbital=0,spin=0))
                    ],
        seqs=       [1,1],
        rcoord=     [0.0,0.0],
        icoord=     [0.0,0.0]
        )
    b=FQuadratic(
        value=      2.0,
        indices=    [   Index(PID(site=0),FID(orbital=0,spin=0)),
                        Index(PID(site=0),FID(orbital=0,spin=0,nambu=CREATION))
                    ],
        seqs=       [0,1],
        rcoord=     [1.0,0.0],
        icoord=     [0.0,0.0]
        )
    print 'a: %s'%a
    print 'a.dagger: %s'%a.dagger
    print 'b: %s'%b
    print 'a.is_Hermitian: %s'%a.is_Hermitian()
    print 'a.is_normal_ordered, a.dagger.is_normal_ordered: %s, %s'%(a.is_normal_ordered(),a.dagger.is_normal_ordered())
    c=FQuadratic(
        value=      1.0j,
        indices=    [   Index(PID(site=1),FID(orbital=0,spin=0)),
                        Index(PID(site=1),FID(orbital=0,spin=0,nambu=CREATION))
                    ],
        seqs=       [1,1],
        rcoord=     [0.0,0.0],
        icoord=     [0.0,0.0],
        )
    d=FQuadratic(
        value=      2.0,
        indices=    [   Index(PID(site=1),FID(orbital=0,spin=0)),
                        Index(PID(site=1),FID(orbital=0,spin=0,nambu=CREATION))
                    ],
        seqs=       [0,1],
        rcoord=     [1.0,0.0],
        icoord=     [0.0,0.0]
        )
    print 'c: %s'%c
    print 'd: %s'%d
    print 'c.id: ',c.id
    print 'd.id: ',d.id
    print 'c+d+c.dagger:\n%s'%(c+d+c.dagger)
    print 'c+(d+2*c):\n%s'%(c+(d+2*c))
    print '(c+d)*2:\n%s'%((c+d)*2)
    print '2*(c+d):\n%s'%(2*(c+d))
    f=Operators()
    f+=c
    f+=c
    f+=d
    print '2*c+d:\n%s'%f
    f*=2.0
    print '4*c+2*d:\n%s'%f
    f-=2*d
    print '4*c:\n%s'%f
    print 'c: %s'%c
    print 'd: %s'%d
    print
