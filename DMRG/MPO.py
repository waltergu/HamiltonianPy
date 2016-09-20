'''
Matrix product operator, including:
1) classes: OptStr
'''

__all__=['OptStr']

from numpy import *
from HamiltonianPy.Basics import OperatorF,OperatorS
from HamiltonianPy.Math.Tensor import *
from MPS import *
import warnings

class OptStr(list):
    '''
    Operator string, which is a special kind of matrix product operator, with the internal indices of the matrices trivial.
        For each of its elements: 2d Tensor
            The matrices of the mpo.
    Attribues:
        labels: set of Label
            The physical labels of the mpo.
    '''

    def __init__(self,ms,labels):
        '''
        Constructor.
        Parameters:
            ms: 2d ndarray
                The matrices of the mpo.
            labels: list of Label
                The second physical labels of the mpo.
        NOTE: For each matrix in the mpo, the first physical label is just the prime of the second.
        '''
        assert len(ms)==len(labels)
        for m,label in zip(ms,labels):
            assert m.ndim==2
            assert isinstance(label,Label)
            self.append(Tensor(m,labels=[label.prime,label]))
        self.labels=set(labels)

    @staticmethod
    def compose(ms):
        '''
        Constructor.
        Parameters:
            ms: list of 2d Tensor.
                The matrices of the mpo.
        Returns: OptStr
            The corresponding optstr.
        '''
        result,labels=OptStr.__new__(OptStr),[]
        for m in ms:
            assert m.ndim==2
            assert m.labels[0]==m.labels[1].prime
            result.append(m)
            labels.append(m.labels[1])
        result.labels=set(labels)
        return result

    @staticmethod
    def from_operator(operator,table):
        '''
        Constructor, which convert an operator to its OptStr form.
        Parameters:
            operator: OperatorS, OperatorF
                The operator which is to be converted to the optstr form.
            table: Table
                The index-sequence table.
        Returns: OptStr
            The corresponding OptStr.
        '''
        if isinstance(operator,OperatorS):
            return opt_str_from_operator_s(operator,table)
        elif isinstance(operator,OperatorF):
            return opt_str_from_operator_f(operator,table)
        else:
            raise ValueError("OptStr.from_operator error: the class of the operator(%s) not supported."%(operator.__class__.__name__))

    def __str__(self):
        '''
        Convert an instance to string.
        '''
        return '\n'.join(str(m) for m in self)

    def matrix(self,us,form='L'):
        '''
        The matrix representation of an optstr on a basis reprented by a mixed matrix product states.
        Parameters:
            us: MPS
                The representing mixed mps of the basis.
            form: 'L' or 'R'
                When 'L', us is left canonical;
                When 'R', us is right canonical.
        Returns: 2d Tensor
            The corresponding matrix representation of the optstr on the basis.
        '''
        table=us.table
        if form=='L':
            start,count=table[self[0].labels[1]],0
            for i,u in enumerate(us[start:]):
                L,S,R=u.labels[MPS.L],u.labels[MPS.S],u.labels[MPS.R]
                up=u.copy(copy_data=False).conjugate()
                if S in self.labels:
                    if i==0:
                        up.relabel(news=[S.prime,R.prime],olds=[S,R])
                        result=contract(up,self[count],u)
                    else:
                        up.relabel(news=[L.prime,S.prime,R.prime],olds=[L,S,R])
                        result=contract(result,up,self[count],u)
                    count+=1
                else:
                    up.relabel(news=[L.prime,R.prime],olds=[L,R])
                    result=contract(result,up,u)
        else:
            end,count=table[self[-1].labels[1]]+1,-1
            for i,u in enumerate(reversed(us[0:end])):
                L,S,R=u.labels[MPS.L],u.labels[MPS.S],u.labels[MPS.R]
                up=u.copy(copy_data=False).conjugate()
                if S in self.labels:
                    if i==0:
                        up.relabel(news=[L.prime,S.prime],olds=[L,S])
                        result=contract(up,self[count],u)
                    else:
                        up.relabel(news=[L.prime,S.prime,R.prime],olds=[L,S,R])
                        result=contract(result,up,self[count],u)
                    count-=1
                else:
                    up.relabel(news=[L.prime,R.prime],olds=[L,R])
                    result=contract(result,up,u)
        return result

    def overlap(self,mps1,mps2):
        '''
        The overlap of an optstr between two mpses.
        Parameters:
            mps1,mps2: MPS
                The two matrix product state between which the overlap of an optstr is calculated.
                Note both mpses are kets, i.e. the complex conjugate of the inner product is taken in this function.
        Returns: number
            The overlap.
        '''
        def reset_and_protect(mps,start):
            if mps.cut==start:
                m,Lambda=mps[mps.cut],mps.Lambda
                mps._reset_(merge='R',reset=None)
            else:
                m,Lambda=mps[mps.cut-1],mps.Lambda
                mps._reset_(merge='L',reset=None)
            return m,Lambda
        assert mps1.table==mps2.table
        if mps1 is mps2:
            start,end,count=mps1.table[self[0].labels[1]],mps1.table[self[-1].labels[1]]+1,0
            if mps1.cut<start or mps1.cut>end:
                warnings.warn("OptStr overlap warning: the cut of the mps is %s and will be moved into the range [%s,%s]."%(mps1.cut,start,end))
                if mps1.cut<start:
                    mps1>>=start-mps1.cut
                else:
                    mps1<<=mps1.cut-end
            m,Lambda=reset_and_protect(mps1,start)
        else:
            start,end,count=0,mps1.nsite,0
            m1,Lambda1=reset_and_protect(mps1,start)
            m2,Lambda2=reset_and_protect(mps2,start)
        result=Tensor(1.0,labels=[])
        for i,(u1,u2) in enumerate(zip(mps1[start:end],mps2[start:end])):
            u1=u1.copy(copy_data=False).conjugate()
            Lp,Sp,Rp=u1.labels[MPS.L],u1.labels[MPS.S],u1.labels[MPS.R]
            L,S,R=u2.labels[MPS.L],u2.labels[MPS.S],u2.labels[MPS.R]
            assert L==Lp and S==Sp and R==Rp
            news,olds=[Lp.prime,Sp.prime,Rp.prime],[Lp,Sp,Rp]
            if i==0:
                news.remove(Lp.prime)
                olds.remove(Lp)
            if i==end-start-1:
                news.remove(Rp.prime)
                olds.remove(Rp)
            if Sp in self.labels:
                u1.relabel(news=news,olds=olds)
                result=contract(result,u1,self[count],u2)
                count+=1
            else:
                news.remove(Sp.prime)
                olds.remove(Sp)
                u1.relabel(news=news,olds=olds)
                result=contract(result,u1,u2)
        if mps1 is mps2:
            mps1._set_ABL_(m,Lambda)
        else:
            mps1._set_ABL_(m1,Lambda1)
            mps2._set_ABL_(m2,Lambda2)
        return asarray(result)

def opt_str_from_operator_s(operator,table):
    '''
    Convert an OperatorS to its corresponding OptStr form.
    Parameters:
        operator: OperatorS
            The operator.
        table: Table
            The index-sequence table.
    Returns: OptStr
        The corresponding optstr.
    '''
    return OptStr.compose(sorted([Tensor(m,labels=[Label(index).prime,Label(index)]) for m,index in zip(operator.spins,operator.indices)],key=lambda key:table[key.labels[1].lb]))

def opt_str_from_operator_f(operator,table):
    '''
    Convert an OperatorF to its corresponding OptStr form.
    Parameters:
        operator: OperatorF
            The operator.
        table: Table
            The index-sequence table.
    Returns: OptStr
        The corresponding optstr.
    '''
    pass
