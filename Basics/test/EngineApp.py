'''
EngineApp test.
'''

__all__=['test_engineapp']

from HamiltonianPy.Basics.DegreeOfFreedom import Status
from HamiltonianPy.Basics.EngineApp import *

def test_engineapp():
    print 'test_engineapp'
    a=Status('Hexagon','CPT')
    a.update({0:1.0})
    a.update({1:2.0+2.0j})
    print a
    a=Engine(din='.',dout='.')
    b=App(name='App',run=lambda engine,app: engine.din)
    a.register(app=b)
    a.register(app=b)
    print
