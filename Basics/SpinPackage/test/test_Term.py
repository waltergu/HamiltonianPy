'''
Spin term test (1 test in total).
'''

__all__=['sterm']

import numpy as np
import numpy.linalg as nl
from HamiltonianPy.Basics.Geometry import *
from HamiltonianPy.Basics.DegreeOfFreedom import *
from HamiltonianPy.Basics.Generator import *
from HamiltonianPy.Basics.SpinPackage import *
from unittest import TestCase,TestLoader,TestSuite

class TestSpinTerm(TestCase):
    def setUp(self):
        print 'test_spin_term'
        J,h=1.0,5.0
        p1,p2=np.array([0.0,0.0]),np.array([1.0,0.0])
        self.lattice=Lattice(name='WG',rcoords=[p1,p2],neighbours=1)
        self.config=IDFConfig(priority=DEFAULT_SPIN_PRIORITY,map=lambda pid: Spin(S=0.5),pids=self.lattice.pids)
        self.terms=[    SpinTerm('J',J,neighbour=1,indexpacks=Heisenberg()),
                        SpinTerm('h',h,neighbour=0,indexpacks=S('z'))
                        ]

    def test_term(self):
        generator=Generator(self.lattice.bonds,self.config,terms=self.terms,dtype=np.float64)
        matrix=0
        for opt in generator.operators:
            matrix+=soptrep(opt,self.config.table())
        result=np.array([
                    [-4.75, 0.0 , 0.0 , 0.0 ],
                    [ 0.0 ,-0.25, 0.5 , 0.0 ],
                    [ 0.0 , 0.5 ,-0.25, 0.0 ],
                    [ 0.0 , 0.0 , 0.0 , 5.25]
                    ])
        self.assertAlmostEqual(nl.norm(matrix.todense()-result),0.0)

sterm=TestSuite([
            TestLoader().loadTestsFromTestCase(TestSpinTerm),
            ])
