'''
Finite DMRG test.
'''

__all__=['test_fdmrg']

from numpy import *
from HamiltonianPy.Basics import *
from HamiltonianPy.DMRG import *

def test_fdmrg():
    print 'test_fdmrg'
    N,J,h=2,0.0,1.0
    # set the lattice.
    points=[]
    a1=array([1.0,0.0])
    for i in xrange(N):
        p1=Point(pid=PID(scope=i,site=0),rcoord=[0.0,0.0]+a1*i,icoord=[0.0,0.0])
        p2=Point(pid=PID(scope=i,site=1),rcoord=[0.5,0.0]+a1*i,icoord=[0.0,0.0])
        points.append(p1)
        points.append(p2)
    lattice=SuperLattice.compose(name='WG',points=points)
    print '\n'.join(["%s:%s"%(pid,point) for pid,point in lattice.items()])
    #lattice.plot(pid_on=True)

    # set the idfconfig
    idfconfig=IDFConfig(priority=DEFAULT_SPIN_PRIORITY)
    for point in points:
        idfconfig[point.pid]=Spin(S=0.5)

    # set the qncconfig
    QuantumNumberCollection.history.clear()
    a=QuantumNumber([('Sz',-1,'U1')])
    b=QuantumNumber([('Sz',1,'U1')])
    qnc=QuantumNumberCollection([(a,1),(b,1)])
    qncconfig=QNCConfig(priority=DEFAULT_SPIN_PRIORITY)
    for index in idfconfig.table():
        qncconfig[index]=qnc

    # set the degfretree.
    layers=(('scope','site','S'),)
    degfres=DegFreTree(qncconfig,layers=layers,priority=DEFAULT_SPIN_PRIORITY)
    for layer in layers:
        print 'layer:',layer
        for i,index in enumerate(degfres.indices(layer)):
            print i,index,degfres[index]
        print
    print

    # set the fdmrg
    fdmrg=fDMRG(
        name=       '1D-spin-1/2',
        lattice=    lattice,
        terms=      [   SpinTerm('J',J,neighbour=1,indexpacks=Heisenberg()),
                        SpinTerm('h',h,neighbour=0,indexpacks=S('WG',matrix=array([[0.5,0],[0,-0.5]])))
                    ],
        config=     idfconfig,
        degfres=     degfres,
        mps=        None
        )
    fdmrg.set_optstrs(layers[0])
    print '\n\n\n'.join(['%s'%(optstr) for optstr in fdmrg.optstrs[layers[0]]['h']])
