'''
Spin operator representation test (1 test in total).
'''

__all__=['soptrep']

import numpy as np
import numpy.linalg as nl
from HamiltonianPy.Basics.Geometry import *
from HamiltonianPy.Basics.DegreeOfFreedom import *
from HamiltonianPy.Basics.Operator import *
from HamiltonianPy.Basics.SpinPackage import *
from HamiltonianPy.Basics.SpinPackage import soptrep as SOPTREP
from unittest import TestCase,TestLoader,TestSuite

class Test_soptrep(TestCase):
    def setUp(self):
        p1,p2=np.array([0.0,0.0]),np.array([1.0,0.0])
        lattice=Lattice(name='WG',rcoords=[p1,p2],neighbours=1)
        config=IDFConfig(priority=DEFAULT_SPIN_PRIORITY,pids=lattice.pids,map=lambda pid: Spin(S=0.5))
        self.table=config.table(mask=[])
        self.opts=Operators()
        for bond in lattice.bonds:
            if bond.neighbour==1:
                spid,epid=bond.spoint.pid,bond.epoint.pid
                sS,eS=config[spid].S,config[epid].S
                sindex,eindex=Index(pid=spid,iid=SID(S=sS)),Index(pid=epid,iid=SID(S=eS))
                self.opts+=SOperator(
                    value=      0.5,
                    indices=    [sindex,eindex],
                    spins=      [SpinMatrix(sS,'p',dtype=np.float64),SpinMatrix(eS,'m',dtype=np.float64)],
                    seqs=       (self.table[sindex],self.table[eindex])
                )
                self.opts+=SOperator(
                    value=      0.5,
                    indices=    [sindex,eindex],
                    spins=      [SpinMatrix(sS,'m',dtype=np.float64),SpinMatrix(eS,'p',dtype=np.float64)],
                    seqs=       (self.table[sindex],self.table[eindex])
                )
                self.opts+=SOperator(
                    value=      1.0,
                    indices=    [sindex,eindex],
                    spins=      [SpinMatrix(sS,'z',dtype=np.float64),SpinMatrix(eS,'z',dtype=np.float64)],
                    seqs=       (self.table[sindex],self.table[eindex])
                    )

    def test_soptrep(self):
        matrix=0.0
        for opt in self.opts.itervalues():
            matrix+=SOPTREP(opt,self.table)
        result=np.array([
                    [ 0.25, 0.0 , 0.0 , 0.0 ],
                    [ 0.0 ,-0.25, 0.5 , 0.0 ],
                    [ 0.0 , 0.5 ,-0.25, 0.0 ],
                    [ 0.0 , 0.0 , 0.0 , 0.25]
                    ])
        self.assertAlmostEqual(nl.norm(matrix.todense()-result),0.0)

soptrep=TestSuite([
            TestLoader().loadTestsFromTestCase(Test_soptrep),
            ])
