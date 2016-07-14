'''
EngineApp test.
'''

__all__=['test_engineapp']

from HamiltonianPy.Basics.EngineApp import *

def test_engineapp():
    print 'test_engineapp'
    a=Name('Hexagon','CPT')
    a.update({0:1.0})
    a.update({1:2.0+2.0j})
    print a
    a=Engine(din='hh',dout='hhh')
    b=App(id='App',run=lambda engine,app: engine.din)
    a.register(app=b)
    a.register(app=b)
    a.runapps()
    a.runapps('App')
    a.runapps('App',clock=True)
    print
