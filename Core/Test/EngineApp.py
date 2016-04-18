from Hamiltonian.Core.BasicClass.EngineAppPy import *
def test_engineapp():
    a=Engine(din='hh',dout='hhh')
    b=App(run=lambda engine,app: engine.din)
    a.addapps(app=b)
    a.addapps('b',b)
    a.runapps()
    a.runapps('App')
    a.runapps('App',clock=True)
