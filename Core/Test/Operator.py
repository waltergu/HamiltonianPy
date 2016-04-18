from Hamiltonian.Core.BasicClass.OperatorPy import *
def test_operator():
    a=Operator(mode='e_quadratic',value=1.0j,indices=[Index(1,0,0,CREATION),Index(1,0,0)],rcoords=[[0.0,0.0],[0.0,0.0]],icoords=[[0,0],[0,0]],seqs=[1,1])
    b=Operator(mode='e_quadratic',value=2.0,indices=[Index(0,0,0),Index(1,0,0,CREATION)],rcoords=[[0.0,0.0],[1.0,0.0]],icoords=[[0,0],[0,0]],seqs=[0,1])
    print a
    print a.dagger
    print b
    print 'a.is_Hermitian:',a.is_Hermitian()
    print 'a.is_combinable(a),a.is_combinable(a.dagger): ',a.is_combinable(a),a.is_combinable(a.dagger)
    print 'a.is_combinable(b): ',a.is_combinable(b)
    print 'a.is_normal_ordered,a.dagger.is_normal_ordered: ',a.is_normal_ordered(),a.dagger.is_normal_ordered()
    c=Operator(mode='e_quadratic',value=1.0j,indices=[Index(1,0,0),Index(1,0,0,CREATION)],rcoords=[[0.0,0.0],[0.0,0.0]],icoords=[[0.0,0.0],[0.0,0.0]],seqs=[1,1])
    d=Operator(mode='e_quadratic',value=2.0,indices=[Index(0,0,0),Index(1,0,0,CREATION)],rcoords=[[0.0,0.0],[1.0,0.0]],icoords=[[0.0,0.0],[0.0,0.0]],seqs=[0,1])    
    print c+d+c.dagger
    print c+(d+2*c)
    print (c+d)*2
    print 2*(c+d)
