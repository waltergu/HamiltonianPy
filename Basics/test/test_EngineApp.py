'''
EngineApp test (1 test in total).
'''

__all__=['engineapp']

from HamiltonianPy.Basics.EngineApp import *
from unittest import TestCase,TestLoader,TestSuite

class TestEngineApp(TestCase):
    def setUp(self):
        self.engine=Engine(din='.',dout='.')
        self.app=App(name='myapp',run=lambda engine,app: engine.din+app.name)

    def test_register(self):
        print()
        self.engine.register(app=self.app)

engineapp=TestSuite([
                    TestLoader().loadTestsFromTestCase(TestEngineApp),
                    ])
