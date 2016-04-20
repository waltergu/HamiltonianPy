from HamiltonianPP.Core.BasicClass.GeometryPy import *
from HamiltonianPP.Core.BasicClass.DegreeOfFreedomPy import *
from HamiltonianPP.Core.BasicClass.OperatorPy import *
from HamiltonianPP.Core.BasicClass.FermionicPackage import *
def test_operator():
    print 'test_operator'
    a=OperatorF(
        mode=       'f_quadratic',
        value=      1.0j,
        indices=    [   Index(PID(site=(0,0,1)),FID(orbital=0,spin=0,nambu=CREATION)),
                        Index(PID(site=(0,0,1)),FID(orbital=0,spin=0))
                    ],
        rcoords=    [[0.0,0.0]],
        icoords=    [[0.0,0.0]],
        seqs=       [1,1]
        )
    b=OperatorF(
        mode=       'f_quadratic',
        value=      2.0,
        indices=    [   Index(PID(site=(0,0,0)),FID(orbital=0,spin=0)),
                        Index(PID(site=(1,0,0)),FID(orbital=0,spin=0,nambu=CREATION))
                    ],
        rcoords=    [[1.0,0.0]],
        icoords=    [[0.0,0.0]],
        seqs=       [0,1]
        )
    print 'a: %s'%a
    print 'a.dagger: %s'%a.dagger
    print 'b: %s'%b
    print 'a.is_Hermitian: %s'%a.is_Hermitian()
    print 'a.is_normal_ordered, a.dagger.is_normal_ordered: %s, %s'%(a.is_normal_ordered(),a.dagger.is_normal_ordered())
    c=OperatorF(
        mode=       'f_quadratic',
        value=      1.0j,
        indices=    [   Index(PID(site=(0,0,1)),FID(orbital=0,spin=0)),
                        Index(PID(site=(0,0,1)),FID(orbital=0,spin=0,nambu=CREATION))
                    ],
        rcoords=    [[0.0,0.0]],
        icoords=    [[0.0,0.0]],
        seqs=       [1,1]
        )
    d=OperatorF(
        mode=       'f_quadratic',
        value=      2.0,
        indices=    [   Index(PID(site=(0,0,1)),FID(orbital=0,spin=0)),
                        Index(PID(site=(0,0,1)),FID(orbital=0,spin=0,nambu=CREATION))
                    ],
        rcoords=    [[1.0,0.0]],
        icoords=    [[0.0,0.0]],
        seqs=       [0,1]
        )
    print 'c: %s'%c
    print 'd: %s'%d
    print 'c.id: ',c.id
    print 'd.id: ',d.id
    print 'c+d+c.dagger:\n%s'%(c+d+c.dagger)
    print 'c+(d+2*c):\n%s'%(c+(d+2*c))
    print '(c+d)*2:\n%s'%((c+d)*2)
    print '2*(c+d):\n%s'%(2*(c+d))
    f=OperatorCollection()
    f+=c
    f+=c
    f+=d
    print '2*c+d: %s'%f
    f*=2.0
    print '4*c+2*d: %s'%f
    f-=2*d
    print '4*c: %s'%f
    print 'c: %s'%c
    print 'd: %s'%d
    print
