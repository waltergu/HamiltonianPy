'''
Lanczos.
'''

__all__=['Lanczos']

from numpy import *
from numpy.linalg import norm
from scipy.sparse import csr_matrix
from scipy.linalg import eigh

class Lanczos:
    '''
    The Lanczos algorithm to deal with csr-formed sparse Hermitian matrices.
    Attributes:
        matrix: csr_matrix
            The csr-formed sparse Hermitian matrix.
        zero: float
            The precision used to cut off the Lanczos iterations.
        new,old: 1D ndarray
            The new and old vectors updated in the Lanczos iterations.
        a,b: 1D list of floats
            The coefficients calculated in the Lanczos iterations.
        cut: logical
            A flag to tag whether the iteration has been cut off.
    '''
    def __init__(self,matrix,vector=None,vtype='rd',zero=10**-10,dtype=complex128):
        '''
        Constructor.
        Parameters:
            matrix: csr_matrix
                The csr-formed sparse Hermitian matrix.
            vector: 1D ndarray,optional
                The initial vector to begin with the Lanczos iterations. 
                It must be normalized already.
            vtype: string,optional
                A flag to tell what type of initial vectors to use when the parameter vector is None.
                'rd' means a random vector while 'sy' means a symmetric vector.
            zero: float,optional
                The precision used to cut off the Lanczos iterations.
            dtype: dtype,optional
                The data type of the iterated vectors.
        '''
        self.matrix=matrix
        self.zero=zero
        if vector is None:
            if vtype.lower()=='rd':
                self.new=zeros(matrix.shape[0],dtype=dtype)
                self.new[:]=random.rand(matrix.shape[0])
            else:
                self.new=ones(matrix.shape[0],dtype=dtype)
            self.new[:]=self.new[:]/norm(self.new)
        else:
            self.new=vector
        self.old=copy(self.new)
        self.cut=False
        self.a=[]
        self.b=[]

    def iter(self):
        '''
        The Lanczos iteration.
        '''
        count=len(self.a)
        buff=self.matrix.dot(self.new)
        self.a.append(vdot(self.new,buff))
        if count>0:
            buff[:]=buff[:]-self.a[count]*self.new-self.b[count-1]*self.old
        else:
            buff[:]=buff[:]-self.a[count]*self.new
        nbuff=norm(buff)
        if nbuff>self.zero:
            self.b.append(nbuff)
            self.old[:]=self.new[:]
            self.new[:]=buff[:]/nbuff
        else:
            self.cut=True
            self.b.append(0.0)
            self.old[:]=self.new[:]
            self.new[:]=0.0

    def tridiagnoal(self):
        '''
        This method returns the tridiagnoal matrix representation of the original sparse Hermitian matrix.
        Returns:
            result: 2D ndarray
                The tridiagnoal matrix representation of the original sparse Hermitian matrix.
        '''
        nmatrix=len(self.a)
        result=zeros((nmatrix,nmatrix))
        for i,(a,b) in enumerate(zip(self.a,self.b)):
            result[i,i]=a.real
            if i<nmatrix-1: 
                result[i+1,i]=b
                result[i,i+1]=b
        return result

    def eig(self,job='n',precision=10**-10):
        '''
        This method returns the ground state energy and optionally the ground state of the original sparse Hermitian matrix.
        Parameters:
            job: string
                A flag to tag what jobs the method does.
                'n' means ground state energy only and 'v' means ground state energy and ground state both.
            precision: float
                The precision of the calculated ground state energy which is used to terminate the Lanczos iteration.
        Returns:
            gse: float
                the ground state energy.
            gs: 1D ndarray,optional
                The ground state. Present when the parameter job is set to be 'V' or 'v'.
        '''
        if job in ('V','v'):gs=copy(self.new)
        delta=1.0;buff=inf
        while not self.cut and delta>precision:
            self.iter()
            if job in ('V','v'):
                w,vs=eigh(self.tridiagnoal())
                gse=w[0];v=vs[:,0]
            else:
                gse=eigh(self.tridiagnoal(),eigvals_only=True)[0]
            delta=abs(gse-buff)
            buff=gse
        if job in ('V','v'):
            self.a=[];self.b=[]
            for i in xrange(len(v)):
                if i==0:
                    self.new[:]=gs[:]
                    gs[:]=0.0
                gs[:]+=self.new*v[i]
                self.iter()
            return gse,gs
        else:
            return gse
