'''
EngineApp test.
'''

__all__=['test_engineapp']

from HamiltonianPy.Basics.EngineApp import *

def test_engineapp():
    print 'test_engineapp'
    a=Engine(din='.',dout='.')
    b=App(name='App',run=lambda engine,app: engine.din)
    a.register(app=b)
    a.register(app=b)
    print
