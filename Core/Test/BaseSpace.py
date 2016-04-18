from Hamiltonian.Core.BasicClass.BaseSpacePy import *
def test_basespace():
    test_kspace()
    test_kspace_functions()
    test_basespace_call()

def test_kspace():
    a=KSpace(reciprocals=[array([2*pi,0.0]),array([0.0,2*pi])],nk=100)
    a.plot(show=True)
    print a.volume['k']/(2*pi)**2
    a=KSpace(reciprocals=[array([1.0,0.0]),array([0.5,sqrt(3.0)/2])],nk=100)
    a.plot(show=True)
    print a.volume
    square_gxm(nk=100).plot()

def test_kspace_functions():
    a=square_bz(reciprocals=[array([1.0,1.0]),array([1.0,-1.0])],nk=100)
    a.plot(show=True)
    print a.volume
    a=rectangle_bz(nk=100)
    a.plot(show=True)
    print a.volume['k']/(2*pi)**2
    a=hexagon_bz(nk=100,vh='v')
    a.plot(show=True)
    print a.volume['k']/(2*pi)**2
    a=hexagon_gkm(nk=100)
    a.plot(show=True)

def test_basespace_call():
    a=BaseSpace(dict(tag='k',mesh=array([1,2,3,4])),{'tag':'t','mesh':array([11,12,13,14])})
    for i,paras in enumerate(a('*')):
        print i,paras
    for i,paras in enumerate(a('+')):
        print i,paras
