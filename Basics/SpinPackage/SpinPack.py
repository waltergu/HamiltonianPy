from numpy import *
from HamiltonianPP.Basics import *
from collections import namedtuple
from scipy.sparse import kron,csr_matrix

__all__=['DEFAULT_SPIN_PRIORITY','SID','Spin','SpinMatrix','OperatorS','SOptRep']

DEFAULT_SPIN_PRIORITY=['socpe','site','S']

class SID(namedtuple('SID',['S'])):
    '''
    '''
    pass

class Spin(Internal):
    '''
    '''
    def __init__(self,S):
        self.S=S

    def __repr__(self):
        '''
        '''
        return 'Spin(S=%s)'%(self.S)

    def __eq__(self,other):
        '''
        Overloaded operator(==).
        '''
        return self.S==other.S

    def table(self,pid):
        return Table([Index(pid=pid,iid=SID(S=self.S))])

class SpinMatrix(ndarray):
    '''
    '''
    def __new__(cls,id,dtype=complex128,**kargs):
        if isinstance(id,tuple):
            delta=lambda i,j: 1 if i==j else 0
            temp=(id[0]/2.0)*(id[0]/2.0+1)
            result=zeros((id[0]+1,id[0]+1),dtype=dtype).view(cls)
            for i in xrange(id[0]+1):
                m=id[0]/2.0-i
                for j in xrange(id[0]+1):
                    n=id[0]/2.0-j
                    if id[1] in ('X','x'):
                        result[i,j]=(delta(i+1,j)+delta(i,j+1))*sqrt(temp-m*n)/2
                    elif id[1] in ('Y','y'):
                        result[i,j]=(delta(i+1,j)-delta(i,j+1))*sqrt(temp-m*n)/(2j)
                    elif id[1] in ('Z','z'):
                        result[i,j]=delta(i,j)*m
                    elif id[1] in ('+'):
                        result[i,j]=delta(i+1,j)*sqrt(temp-m*n)
                    elif id[1] in ('-'):
                        result[i,j]=delta(i,j+1)*sqrt(temp-m*n)
                    else:
                        raise ValueError('SpinMatrix construction error: id=%s not supported.'%(id,))
            result.id=id
        else:
            raise ValueError('SpinMatrix construction error: id must be a tuple.')
        return result

    def __str__(self):
        return "SpinMatrix(id=%s,\nmatrix=\n%s\n)"%(self.id,super(SpinMatrix,self).__str__())

class OperatorS(Operator):
    '''
    '''
    def __init__(self,value,indices,spins,rcoords,icoords,seqs):
        self.value=value
        self.indices=indices
        self.spins=spins
        self.rcoords=rcoords
        self.icoords=icoords
        self.seqs=seqs
        self.set_id()

    def set_id(self):
        self.id=tuple(list(self.indices)+[spin.id for spin in self.spins])

    def __repr__(self):
        return 'OperatorS(value=%s, indices=%s, spins=%s, rcoords=%s, icoords=%s, seqs=%s)'%(self.value,self.indices,self.spins,self.rcoords,self.icoords,self.seqs)

def SOptRep(operator,table):
    temp=[eye(index.S+1) for index in sorted(table.keys(),key=table.get)]
    for spin,seq in zip(operator.spins,operator.seqs):
        temp[seq]=asarray(spin)
    result=operator.value
    for matrix in temp:
        result=kron(result,matrix,format='csr')
        result.eliminate_zeros()
    return result

def test():
    test_spin_matrix()

def test_spin_matrix():
    N=2
    print SpinMatrix((N,'x'),dtype=float64)
    print SpinMatrix((N,'y'),dtype=complex128)
    print SpinMatrix((N,'z'),dtype=float64)
    print SpinMatrix((N,'+'),dtype=float64)
    print SpinMatrix((N,'-'),dtype=float64)

if __name__=='__main__':
    test()
